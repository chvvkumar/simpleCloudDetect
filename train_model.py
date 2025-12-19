import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import random
import copy
import time
from PIL import Image

# ---------------------------------------------------------
# Custom Dataset to handle recursive files + splits
# ---------------------------------------------------------
class CloudDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]
        try:
            image = Image.open(path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"⚠️ Warning: Corrupt image at {path}, skipping...")
            # Simple fallback: return the next image or previous one
            return self.__getitem__((idx + 1) % len(self))

def find_images_recursively(root_dir):
    """Scans directory recursively and returns paths, labels, and class mapping."""
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    images = []
    labels = []
    
    for cls_name in classes:
        class_dir = os.path.join(root_dir, cls_name)
        class_idx = class_to_idx[cls_name]
        
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    images.append(os.path.join(root, file))
                    labels.append(class_idx)
                    
    return images, labels, classes

# ---------------------------------------------------------
# Main Training Function
# ---------------------------------------------------------
def train_model(data_dir, output_model='model.onnx', output_labels='labels.txt', 
                epochs=50, batch_size=256, patience=7, num_workers=8, arch='mobilenet_v3_large'):
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Architecture: {arch}")
        # Reset peak memory stats at start
        torch.cuda.reset_peak_memory_stats()
        # Optimization for consistent input sizes
        torch.backends.cudnn.benchmark = True

    # 2. Prepare Data (Find -> Split -> Transform)
    if not os.path.exists(data_dir):
        print(f"❌ Error: '{data_dir}' not found.")
        return

    print("🔍 Scanning images recursively...")
    all_paths, all_labels, class_names = find_images_recursively(data_dir)
    
    if len(all_paths) == 0:
        print("❌ No images found!")
        return

    # Shuffle and Split (80% Train, 20% Val)
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    split_idx = int(0.8 * len(all_paths))
    train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
    
    print(f"📂 Data Split: {len(train_paths)} Training, {len(val_paths)} Validation")
    print(f"🏷️  Classes: {class_names}")

    # Transforms
    # Train: Heavy augmentation to prevent overfitting
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), # Sky looks similar upside down sometimes
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Val: Clean (just resize and normalize)
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets & Loaders
    train_dataset = CloudDataset(train_paths, train_labels, transform=train_transforms)
    val_dataset = CloudDataset(val_paths, val_labels, transform=val_transforms)

    # Pin_memory=True speeds up CPU->GPU transfer
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                            num_workers=num_workers, pin_memory=True)

    # 3. Model Setup (Architecture Selection)
    print(f"🏗️  Initializing {arch}...")
    
    if arch == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        # Modify classifier for specific number of classes
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
        
    elif arch == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        # Modify classifier for specific number of classes
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
        
    elif arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # ResNet uses 'fc' (fully connected) instead of 'classifier'
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        
    else:
        print(f"❌ Error: Architecture '{arch}' not supported.")
        return

    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Scheduler: Drop learning rate if validation loss stalls
    # Removed verbose=True as it is deprecated/removed in newer PyTorch versions
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # 4. Training Loop with Early Stopping
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n🔥 Starting Training Loop...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 10)

        # Reset peak memory for this epoch
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Timing Stats
        total_data_time = 0.0
        total_compute_time = 0.0
        num_batches = len(train_loader)
        end_time = time.time()

        try:
            for i, (inputs, labels) in enumerate(train_loader):
                # Measure Data Loading Time
                data_time = time.time() - end_time
                total_data_time += data_time

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Start Compute Timing
                compute_start = time.time()

                optimizer.zero_grad()

                # Enable mixed precision could speed up more, but let's stick to simple FP32 with high batch size first
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()
                
                # Measure Compute Time
                compute_time = time.time() - compute_start
                total_compute_time += compute_time

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Reset end time
                end_time = time.time()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n🛑 CUDA Out of Memory! Batch size {batch_size} is too large.")
                print("   Try reducing it to 256 or 384.")
                if device.type == 'cuda':
                    print(f"   Peak Memory reached: {torch.cuda.max_memory_allocated(0)/1024**3:.2f}GB")
                    torch.cuda.empty_cache()
                return
            else:
                raise e

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Timing Analysis
        avg_data_time = total_data_time / num_batches
        avg_compute_time = total_compute_time / num_batches
        print(f"   Time (Avg/Batch): Data: {avg_data_time*1000:.1f}ms | Compute: {avg_compute_time*1000:.1f}ms")
        
        if avg_data_time > avg_compute_time:
             print("   ⚠️  Bottleneck: DATA LOADING (CPU/Disk). Increase num_workers or batch_size.")
        else:
             print("   ✅ Bottleneck: COMPUTE (GPU). This is ideal.")

        # GPU Monitoring (Peak vs Current)
        if device.type == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(0) / 1024**3
            cur_mem = torch.cuda.memory_allocated(0) / 1024**3
            print(f"   GPU Mem: Peak: {max_mem:.2f}GB | Current: {cur_mem:.2f}GB")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_epoch_loss = val_loss / len(val_dataset)
        val_epoch_acc = val_corrects.double() / len(val_dataset)
        print(f"Val   Loss: {val_epoch_loss:.4f} Acc: {val_epoch_acc:.4f}")
        
        # Step Scheduler
        scheduler.step(val_epoch_loss)

        # --- Early Stopping Logic ---
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("✅ Validation loss improved, saving model state...")
        else:
            epochs_no_improve += 1
            print(f"⚠️ No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("🛑 Early stopping triggered!")
                break

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")

    # 5. Export Best Model
    print("📦 Exporting BEST model to ONNX...")
    model.load_state_dict(best_model_wts)
    model.eval()
    
    # Create dummy input for export
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(model, dummy_input, output_model, 
                      input_names=['input'], output_names=['output'],
                      opset_version=12)
    
    # Save Labels
    with open(output_labels, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
            
    print(f"🎉 Done! Model saved to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/f/MLClouds_incoming/resized/', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs (will likely stop early)')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size (Higher = Better GPU Usage)')
    parser.add_argument('--patience', type=int, default=7, help='Epochs to wait before early stopping')
    parser.add_argument('--workers', type=int, default=8, help='Number of data loader workers')
    parser.add_argument('--arch', type=str, default='mobilenet_v3_large', 
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18'],
                        help='Model architecture: mobilenet_v3_small, mobilenet_v3_large, or resnet18')
    
    args = parser.parse_args()
    
    train_model(args.data_dir, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                patience=args.patience,
                num_workers=args.workers,
                arch=args.arch)