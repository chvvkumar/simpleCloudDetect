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
import sys
import math

# ---------------------------------------------------------
# Helper: Check Available Memory in WSL/Linux
# ---------------------------------------------------------
def get_available_memory_gb():
    try:
        with open('/proc/meminfo', 'r') as f:
            for line in f:
                if 'MemAvailable' in line:
                    # format: MemAvailable:    16300000 kB
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024) # Convert to GB
    except:
        return 1.0 # Fail safe
    return 1.0

# ---------------------------------------------------------
# Custom Dataset to handle recursive files + splits + RAM limit
# ---------------------------------------------------------
class CloudDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, cache_size_gb=0, pre_resize=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache_limit_bytes = cache_size_gb * (1024**3)
        self.cached_images = [None] * len(image_paths)
        self.pre_resize = pre_resize

        if self.cache_limit_bytes > 0:
            print(f"🧠 Caching images to RAM (Target: {cache_size_gb:.2f} GB)...")
            current_bytes = 0
            count = 0
            
            for i, path in enumerate(image_paths):
                try:
                    with Image.open(path) as img:
                        # OPTIMIZATION: Resize BEFORE caching
                        # This saves massive amounts of RAM and CPU time during training
                        loaded_img = img.convert('RGB')
                        
                        if self.pre_resize:
                            # Resize to target size (e.g. 224x224) immediately
                            loaded_img = loaded_img.resize((self.pre_resize, self.pre_resize), Image.Resampling.BILINEAR)

                        # Estimate size: W * H * 3 bytes
                        est_size = loaded_img.size[0] * loaded_img.size[1] * 3
                        
                        if current_bytes + est_size > self.cache_limit_bytes:
                            if count == 0:
                                print("⚠️  Warning: Cache limit too low to store even one image.")
                            break
                        
                        # Force load into memory
                        loaded_img.load()
                        self.cached_images[i] = loaded_img
                        current_bytes += est_size
                        count += 1
                        
                    if count % 5000 == 0 and count > 0:
                        print(f"   Cached {count} images ({current_bytes / (1024**3):.2f} GB)...")
                        
                except Exception as e:
                    print(f"⚠️ Warning: Could not cache {path}: {e}")
            
            usage_gb = current_bytes / (1024**3)
            percent = (count / len(image_paths)) * 100
            print(f"✅ Cache complete. Used {usage_gb:.2f} GB. Cached {count}/{len(image_paths)} images ({percent:.1f}%).")
            if count < len(image_paths):
                print("   (Remaining images will be loaded from disk on-the-fly)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Retry loop to handle corrupt images without recursion depth errors
        attempt = 0
        max_attempts = 10
        current_idx = idx

        while attempt < max_attempts:
            label = self.labels[current_idx]
            
            image = None
            
            # 1. Try RAM Cache
            if self.cached_images[current_idx] is not None:
                image = self.cached_images[current_idx]
            else:
                # 2. Disk Fallback
                path = self.image_paths[current_idx]
                try:
                    with Image.open(path) as img:
                        image = img.convert('RGB')
                        # Note: We don't pre-resize disk images here as transforms handle it
                except Exception as e:
                    pass

            if image is not None:
                if self.transform:
                    image = self.transform(image)
                return image, label
            
            # Failed, try next image
            current_idx = (current_idx + 1) % len(self)
            attempt += 1

        # If all fails, return a blank tensor to prevent crash
        print(f"❌ Critical: Could not find valid image after {max_attempts} attempts.")
        return torch.zeros(3, 224, 224), 0

def find_images_recursively(root_dir):
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
                epochs=50, batch_size=256, patience=7, num_workers=8, arch='mobilenet_v3_large',
                cache_ram_gb=0.0):
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    
    # --- AUTO-SAFETY: Check Memory ---
    if cache_ram_gb > 0:
        available_gb = get_available_memory_gb()
        safe_limit = available_gb * 0.7  # Leave 30% buffer for OS + Training
        
        if cache_ram_gb > safe_limit:
            print(f"⚠️  WARNING: You requested {cache_ram_gb} GB cache, but only {available_gb:.1f} GB is free.")
            print(f"   Auto-clamping cache to safe limit: {safe_limit:.1f} GB")
            cache_ram_gb = safe_limit
            
        # --- AUTO-OPTIMIZATION: Workers ---
        # If data is in RAM, multiple workers just waste memory. Set to 0 (main process).
        if num_workers > 0:
            print("ℹ️  Optimization: RAM Cache enabled. Setting num_workers=0 to save memory.")
            num_workers = 0

    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Architecture: {arch}")
        print(f"   RAM Cache: {cache_ram_gb:.1f} GB")
        torch.cuda.reset_peak_memory_stats()
        torch.backends.cudnn.benchmark = False

    # 2. Prepare Data
    if not os.path.exists(data_dir):
        print(f"❌ Error: '{data_dir}' not found.")
        return

    print("🔍 Scanning images recursively...")
    all_paths, all_labels, class_names = find_images_recursively(data_dir)
    
    if len(all_paths) == 0:
        print("❌ No images found!")
        return

    # Check Image Size on a sample
    try:
        sample_img = Image.open(all_paths[0])
        print(f"📏 Sample Image Size: {sample_img.size}")
    except:
        pass

    # Shuffle and Split
    combined = list(zip(all_paths, all_labels))
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    # 80/20 Split
    split_idx = int(0.8 * len(all_paths))
    train_paths, val_paths = all_paths[:split_idx], all_paths[split_idx:]
    train_labels, val_labels = all_labels[:split_idx], all_labels[split_idx:]
    
    print(f"📂 Data Split: {len(train_paths)} Training, {len(val_paths)} Validation")
    print(f"🏷️  Classes: {class_names}")

    # Calculate Cache Budgets
    train_cache_gb = cache_ram_gb * 0.8
    val_cache_gb = cache_ram_gb * 0.2

    # Transforms
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(), 
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    if cache_ram_gb > 0:
        print(f"🧠 Initializing Datasets (Max RAM: {cache_ram_gb:.1f} GB) with Pre-Resize=224px...")
    
    # Pass pre_resize=224 to shrink images in RAM
    train_dataset = CloudDataset(train_paths, train_labels, transform=train_transforms, 
                                 cache_size_gb=train_cache_gb, pre_resize=224)
    val_dataset = CloudDataset(val_paths, val_labels, transform=val_transforms, 
                               cache_size_gb=val_cache_gb, pre_resize=224)

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True,
        'persistent_workers': True if num_workers > 0 else False,
        'prefetch_factor': 2 if num_workers > 0 else None
    }
    
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

    # 3. Model Setup
    print(f"🏗️  Initializing {arch}...")
    
    if arch == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    elif arch == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    elif arch == 'resnet18':
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    elif arch == 'efficientnet_b0':
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(class_names))
    else:
        print(f"❌ Error: Architecture '{arch}' not supported.")
        return

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    # 4. Training Loop
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    
    print("\n🔥 Starting Training Loop...")
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 10)

        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats()

        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        total_data_time = 0.0
        total_compute_time = 0.0
        num_batches = len(train_loader)
        end_time = time.time()

        try:
            for i, (inputs, labels) in enumerate(train_loader):
                data_time = time.time() - end_time
                total_data_time += data_time

                inputs = inputs.to(device)
                labels = labels.to(device)
                
                compute_start = time.time()
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                compute_time = time.time() - compute_start
                total_compute_time += compute_time

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                # Progress Update every 10 batches
                if i % 10 == 0:
                    print(f"   Step {i}/{num_batches} | Data: {data_time*1000:.1f}ms | GPU: {compute_time*1000:.1f}ms")

                end_time = time.time()
        
        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"\n🛑 CUDA Out of Memory! Batch size {batch_size} is too large.")
                return
            elif "unable to find an engine" in str(e):
                print(f"\n🛑 CuDNN Benchmark Error!")
                return
            else:
                raise e

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # Timing Analysis
        avg_data_time = total_data_time / num_batches
        avg_compute_time = total_compute_time / num_batches
        print(f"   ⏱️  Avg Timings: Data: {avg_data_time*1000:.1f}ms | GPU: {avg_compute_time*1000:.1f}ms")
        
        if avg_data_time > avg_compute_time:
             print("   ⚠️  BOTTLENECK: CPU/Disk. (If RAM cache is on, CPU transforms are slow)")
        else:
             print("   ✅ BOTTLENECK: GPU (Optimal)")

        if device.type == 'cuda':
            max_mem = torch.cuda.max_memory_allocated(0) / 1024**3
            print(f"   💾 GPU Mem: {max_mem:.2f} GB Peak")

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
        
        scheduler.step(val_epoch_loss)

        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("   ⭐ New Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"   ⚠️ No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("🛑 Early stopping triggered!")
                break

    print(f"\nTraining complete. Best Val Acc: {best_acc:.4f}")

    print("📦 Exporting BEST model to ONNX (Opset 18)...")
    model.load_state_dict(best_model_wts)
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    # UPDATED: Using opset_version 18 to support PyTorch Nightly
    torch.onnx.export(model, dummy_input, output_model, 
                      input_names=['input'], output_names=['output'],
                      opset_version=18)
    
    with open(output_labels, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
            
    print(f"🎉 Done! Model saved to {output_model}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='/mnt/f/MLClouds_incoming/resized/', help='Path to dataset')
    parser.add_argument('--epochs', type=int, default=50, help='Max epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--patience', type=int, default=7, help='Early stopping patience')
    parser.add_argument('--workers', type=int, default=8, help='Data loader workers')
    parser.add_argument('--arch', type=str, default='mobilenet_v3_large', 
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18', 'efficientnet_b0'])
    # Changed from boolean switch to float for GB limit
    parser.add_argument('--cache_ram', type=float, default=0.0, help='Max RAM to use for caching images in GB (0 = disabled)')
    
    args = parser.parse_args()
    
    train_model(args.data_dir, 
                epochs=args.epochs, 
                batch_size=args.batch_size, 
                patience=args.patience,
                num_workers=args.workers,
                arch=args.arch,
                cache_ram_gb=args.cache_ram)