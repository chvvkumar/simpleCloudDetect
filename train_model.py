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
import sys
import gc
from PIL import Image, ImageFile

# Ensure truncated images don't crash the script
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ---------------------------------------------------------
# Custom Dataset with RAM Caching & Recursive Search
# ---------------------------------------------------------
class CloudDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None, cache_ram_gb=0):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.cache = {}
        self.use_cache = cache_ram_gb > 0
        
        # Initialize RAM Cache
        if self.use_cache:
            self._preload_images(cache_ram_gb)

    def _preload_images(self, limit_gb):
        """Pre-loads images into RAM up to the specified GB limit."""
        print(f"🧠 Caching images into RAM (Limit: {limit_gb} GB)...")
        limit_bytes = limit_gb * (1024 ** 3)
        current_bytes = 0
        count = 0
        
        # Shuffle paths to cache a random sample if we can't fit all
        indices = list(range(len(self.image_paths)))
        # We don't shuffle here to keep index alignment simple, 
        # but we iterate sequentially. 
        
        for idx in indices:
            path = self.image_paths[idx]
            try:
                # Open and force load into memory
                img = Image.open(path).convert('RGB')
                img.load() 
                
                # Estimate size: H * W * 3 channels (approximate)
                img_size = img.size[0] * img.size[1] * 3
                
                if current_bytes + img_size > limit_bytes:
                    print(f"   ⚠️ RAM Cache full! Cached {count}/{len(self.image_paths)} images.")
                    break
                
                self.cache[idx] = img
                current_bytes += img_size
                count += 1
            except Exception as e:
                pass # Skip corrupt files during pre-load

        print(f"   ✅ Cached {count} images ({current_bytes / (1024**3):.2f} GB used)")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Check Cache
        if idx in self.cache:
            image = self.cache[idx]
        else:
            # 2. Load from Disk
            path = self.image_paths[idx]
            try:
                image = Image.open(path).convert('RGB')
            except Exception as e:
                print(f"⚠️ Warning: Corrupt image at {path}, skipping...")
                # Fallback to a random image to maintain batch size
                new_idx = random.randint(0, len(self) - 1)
                return self.__getitem__(new_idx)

        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

def find_images_recursively(root_dir):
    """
    Scans directory recursively.
    Assumes structure: root_dir/class_name/subfolders/image.jpg
    It will find images no matter how deep they are inside the class folder.
    """
    # Get immediate subdirectories as class names
    classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    images = []
    labels = []
    
    for cls_name in classes:
        class_dir = os.path.join(root_dir, cls_name)
        class_idx = class_to_idx[cls_name]
        
        # os.walk goes deep into all sub-levels (Dates, Sessions, etc.)
        for root, _, files in os.walk(class_dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
                    images.append(os.path.join(root, file))
                    labels.append(class_idx)
                    
    return images, labels, classes

# ---------------------------------------------------------
# Training Routine
# ---------------------------------------------------------
def train_model(data_dir, output_model, output_labels, epochs, batch_size, 
                patience, num_workers, arch, cache_ram):
    
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🚀 Device: {device}")
    
    if device.type == 'cuda':
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Architecture: {arch}")
        # RTX 50-Series / Newer CuDNN Fix: Disable benchmarking if it crashes
        torch.backends.cudnn.benchmark = False 
        torch.cuda.reset_peak_memory_stats()

    # 2. Prepare Data
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Directory '{data_dir}' not found.")

    print("🔍 Scanning images recursively (including subfolders)...")
    all_paths, all_labels, class_names = find_images_recursively(data_dir)
    
    if len(all_paths) == 0:
        raise ValueError("No images found! Check your directory structure.")

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
    # Pass cache_ram only to training set usually, or split budget. 
    # Here we give the full budget to the training set for max speed.
    train_dataset = CloudDataset(train_paths, train_labels, transform=train_transforms, cache_ram_gb=cache_ram)
    val_dataset = CloudDataset(val_paths, val_labels, transform=val_transforms, cache_ram_gb=0) # Don't cache val

    loader_kwargs = {
        'batch_size': batch_size,
        'num_workers': num_workers,
        'pin_memory': True if device.type == 'cuda' else False,
        'persistent_workers': True if num_workers > 0 else False,
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
    else:
        raise ValueError(f"Architecture {arch} not supported.")

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
        print("-" * 20)

        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Performance Monitoring
        total_data_time = 0.0
        total_compute_time = 0.0
        num_batches = len(train_loader)
        end_time = time.time()

        for i, (inputs, labels) in enumerate(train_loader):
            # 1. Data Loading Time
            data_time = time.time() - end_time
            total_data_time += data_time

            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # 2. Compute Time
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
            
            # Update end_time for next iteration
            end_time = time.time()

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = running_corrects.double() / len(train_dataset)
        
        # --- Bottleneck Analysis ---
        avg_data_ms = (total_data_time / num_batches) * 1000
        avg_comp_ms = (total_compute_time / num_batches) * 1000
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")
        print(f"   ⏱️  Timings: Data: {avg_data_ms:.1f}ms | GPU: {avg_comp_ms:.1f}ms")
        
        if avg_data_ms > avg_comp_ms:
            print("   ⚠️  BOTTLENECK: CPU/Disk. Increase --workers or --cache_ram.")
        else:
            print("   ✅ BOTTLENECK: GPU. System is optimized.")

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

        # --- Early Stopping ---
        if val_epoch_loss < best_loss:
            best_loss = val_epoch_loss
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
            print("   ⭐ New Best Model Saved!")
        else:
            epochs_no_improve += 1
            print(f"   ⏳ No improvement for {epochs_no_improve}/{patience} epochs.")
            if epochs_no_improve >= patience:
                print("🛑 Early stopping triggered!")
                break

    # 5. Export
    print(f"\n📦 Exporting Best Model (Acc: {best_acc:.4f})...")
    model.load_state_dict(best_model_wts)
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    torch.onnx.export(model, dummy_input, output_model, 
                      input_names=['input'], output_names=['output'],
                      opset_version=12)
    
    with open(output_labels, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
            
    print(f"🎉 Success! Saved to {output_model}")


# ---------------------------------------------------------
# Main Execution with Smart Fallbacks
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    parser.add_argument('--output_model', type=str, default='model.onnx')
    parser.add_argument('--output_labels', type=str, default='labels.txt')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--cache_ram', type=float, default=0.0, help='Amount of RAM in GB to use for caching images')
    parser.add_argument('--arch', type=str, default='mobilenet_v3_large', 
                        choices=['mobilenet_v3_small', 'mobilenet_v3_large', 'resnet18'])
    
    args = parser.parse_args()

    # Smart Fallback Loop
    # If OOM occurs, we reduce batch size and restart
    current_batch_size = args.batch_size
    
    while current_batch_size >= 4:
        try:
            train_model(
                data_dir=args.data_dir,
                output_model=args.output_model,
                output_labels=args.output_labels,
                epochs=args.epochs,
                batch_size=current_batch_size,
                patience=args.patience,
                num_workers=args.workers,
                arch=args.arch,
                cache_ram=args.cache_ram
            )
            break # If successful, exit loop

        except RuntimeError as e:
            error_msg = str(e)
            if "out of memory" in error_msg:
                print(f"\n💥 CUDA OOM Error with batch size {current_batch_size}!")
                print("   ♻️  Clearing cache and retrying with smaller batch size...")
                
                # Clear memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
                
                # Halve batch size
                current_batch_size //= 2
                
            elif "cuDNN" in error_msg or "find an engine" in error_msg:
                print(f"\n💥 CuDNN Error with batch size {current_batch_size}!")
                print("   ♻️  This is common on RTX 30/40/50 series. Retrying with smaller batch...")
                current_batch_size //= 2
                
            else:
                # Re-raise other errors
                raise e
    
    if current_batch_size < 4:
        print("\n❌ Failed to train even with batch size 4. Your GPU memory might be too small for this model/image size.")