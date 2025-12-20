import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import os
import argparse
import random
import copy
import time
import datetime
import math

# --- DALI IMPORTS ---
from nvidia.dali import pipeline_def, fn, types
from nvidia.dali.plugin.pytorch import DALIGenericIterator, LastBatchPolicy

# ---------------------------------------------------------
# Helper: Find Images
# ---------------------------------------------------------
def find_images_recursively(root_dir):
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    images = []
    labels = []
    
    print(f"🔍 Scanning '{root_dir}'...")
    for cls_name in classes:
        class_dir = os.path.join(root_dir, cls_name)
        class_idx = class_to_idx[cls_name]
        
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    abs_path = os.path.abspath(os.path.join(root, file))
                    images.append(abs_path)
                    labels.append(class_idx)
                    
    return images, labels, classes

def save_file_list(paths, labels, filename):
    with open(filename, 'w') as f:
        for path, label in zip(paths, labels):
            f.write(f"{path} {label}\n")

# ---------------------------------------------------------
# DALI Pipeline (CPU ONLY for WSL2 Stability)
# ---------------------------------------------------------
@pipeline_def
def create_dali_pipeline(file_list_path, is_training):
    jpegs, labels = fn.readers.file(
        file_list=file_list_path,
        random_shuffle=is_training,
        name="Reader"
    )
    
    # 1. Decode on CPU
    images = fn.decoders.image(jpegs, device="cpu", output_type=types.RGB)
    
    # 2. Resize on CPU
    images = fn.resize(images, resize_x=300, resize_y=300)

    # 3. Augmentations (CPU)
    if is_training:
        images = fn.flip(images, 
                         horizontal=fn.random.coin_flip(probability=0.5),
                         vertical=fn.random.coin_flip(probability=0.5))
        images = fn.rotate(images, angle=fn.random.uniform(range=(-20, 20)))
        images = fn.color_twist(images, 
                                brightness=fn.random.uniform(range=(0.8, 1.2)),
                                contrast=fn.random.uniform(range=(0.8, 1.2)))

    # 4. Normalize (CPU)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_layout="CHW"
    )
    
    labels = fn.cast(labels, dtype=types.INT64)
    return images, labels

# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------
def train_model(data_dir, output_model='model.onnx', output_labels='labels.txt', 
                epochs=50, batch_size=256, patience=7, arch='mobilenet_v3_large'):
    
    total_start_time = time.time()
    
    # Setup Data
    if not os.path.exists(data_dir):
        print(f"❌ Error: '{data_dir}' not found.")
        return

    all_paths, all_labels, class_names = find_images_recursively(data_dir)
    if not all_paths: return

    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    split_idx = int(0.8 * len(all_paths))
    train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]

    print("📝 Generating DALI file lists...")
    save_file_list(train_paths, train_labels, "dali_train.txt")
    save_file_list(val_paths, val_labels, "dali_val.txt")
    print(f"📂 Data Split: {len(train_paths)} Training, {len(val_paths)} Validation")

    # Build DALI Loaders
    print("🚀 Initializing DALI Pipelines (CPU Mode + Dynamic Shape)...")
    
    pipe_train = create_dali_pipeline(file_list_path="dali_train.txt", is_training=True, batch_size=batch_size, num_threads=4, device_id=0)
    pipe_train.build()
    
    pipe_val = create_dali_pipeline(file_list_path="dali_val.txt", is_training=False, batch_size=batch_size, num_threads=4, device_id=0)
    pipe_val.build()

    # --- KEY FIX: dynamic_shape=True allows non-contiguous CPU memory ---
    train_loader = DALIGenericIterator(
        pipe_train, ['data', 'label'], reader_name="Reader", 
        last_batch_policy=LastBatchPolicy.DROP, auto_reset=True,
        dynamic_shape=True 
    )
    
    val_loader = DALIGenericIterator(
        pipe_val, ['data', 'label'], reader_name="Reader", 
        last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True,
        dynamic_shape=True
    )

    # Model Setup
    print(f"🏗️  Initializing {arch}...")
    device = torch.device("cuda")
    
    if arch == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    elif arch == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n🔥 Starting Training Loop...")

    try:
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 10)

            # --- Training ---
            model.train()
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            start_time = time.time()
            
            for i, data in enumerate(train_loader):
                # --- KEY FIX: Manually stack list of tensors and move to GPU ---
                # data[0]["data"] is a list of CPU tensors. We stack them into one batch.
                inputs = torch.stack(data[0]["data"]).to(device)
                
                # Labels come as a list of 1-element tensors, flatten and stack
                labels = torch.stack(data[0]["label"]).squeeze().long().to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                batch_len = inputs.size(0)
                running_loss += loss.item() * batch_len
                running_corrects += torch.sum(preds == labels)
                total_samples += batch_len

            epoch_time = time.time() - start_time
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Time: {epoch_time:.1f}s")
            print(f"   🚀 Speed: {total_samples/epoch_time:.0f} images/sec")

            # --- Validation ---
            model.eval()
            val_loss = 0.0
            val_corrects = 0
            val_samples = 0

            with torch.no_grad():
                for data in val_loader:
                    inputs = torch.stack(data[0]["data"]).to(device)
                    labels = torch.stack(data[0]["label"]).squeeze().long().to(device)
                    
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    val_corrects += torch.sum(preds == labels)
                    val_samples += inputs.size(0)

            val_epoch_loss = val_loss / val_samples
            val_epoch_acc = val_corrects.double() / val_samples
            print(f"Val   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
            
            scheduler.step(val_epoch_loss)

            if val_epoch_loss < best_loss:
                best_loss = val_epoch_loss
                best_acc = val_epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
                print("   ⭐ New Best Model Saved!")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print("🛑 Early stopping triggered!")
                    break

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
    
    # Export
    print(f"\n🏆 Best Val Acc: {best_acc:.4f}")
    print("📦 Exporting BEST model to ONNX...")
    model.load_state_dict(best_model_wts)
    model.eval()
    dummy_input = torch.randn(1, 3, 300, 300, device=device)
    torch.onnx.export(model, dummy_input, output_model, input_names=['input'], output_names=['output'], opset_version=18)
    
    with open(output_labels, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
            
    if os.path.exists("dali_train.txt"): os.remove("dali_train.txt")
    if os.path.exists("dali_val.txt"): os.remove("dali_val.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--arch', type=str, default='mobilenet_v3_large')
    args = parser.parse_args()
    
    train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, arch=args.arch)