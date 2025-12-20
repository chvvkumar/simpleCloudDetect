import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import time
import copy
import datetime
from PIL import Image

# ---------------------------------------------------------
# 1. Custom Dataset with Pre-Resize RAM Caching
# ---------------------------------------------------------
class CloudDataset(Dataset):
    def __init__(self, file_list, labels, cache_ram=False):
        self.file_list = file_list
        self.labels = labels
        self.cache_ram = cache_ram
        self.cached_images = []
        
        # Standardize size here (CPU only does this ONCE)
        self.pre_process = transforms.Compose([
            transforms.Resize((300, 300)),
            transforms.ToTensor(), # Converts to [0.0, 1.0] float
        ])

        if self.cache_ram:
            print(f"🧠 Caching {len(file_list)} images to RAM (Resized to 300x300)...")
            for idx, path in enumerate(file_list):
                try:
                    with Image.open(path).convert('RGB') as img:
                        # Store as lightweight tensor
                        tensor = self.pre_process(img)
                        self.cached_images.append(tensor)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {path}: {e}")
                    # Placeholder for bad image (black)
                    self.cached_images.append(torch.zeros(3, 300, 300))
                    
                if (idx + 1) % 5000 == 0:
                    print(f"   Cached {idx + 1} images...")
            print("✅ Cache complete.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.cache_ram:
            image = self.cached_images[idx]
        else:
            # Fallback for disk reading (slow)
            path = self.file_list[idx]
            try:
                with Image.open(path).convert('RGB') as img:
                    image = self.pre_process(img)
            except:
                image = torch.zeros(3, 300, 300)
        
        return image, label

# ---------------------------------------------------------
# 2. GPU Augmentation Module
# ---------------------------------------------------------
class GPUAugment(nn.Module):
    def __init__(self):
        super().__init__()
        # These run on the GPU batch [B, 3, 300, 300]
        self.transforms = nn.Sequential(
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(20),
            # Normalize (ImageNet stats)
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                 std=[0.229, 0.224, 0.225])
        )
        
        self.normalize_only = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, x, is_training=True):
        if is_training:
            return self.transforms(x)
        else:
            return self.normalize_only(x)

# ---------------------------------------------------------
# 3. Helper: Find Images
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
                    images.append(os.path.join(root, file))
                    labels.append(class_idx)
    return images, labels, classes

# ---------------------------------------------------------
# 4. Main Training Function
# ---------------------------------------------------------
def train_model(data_dir, output_model='model.onnx', output_labels='labels.txt', 
                epochs=50, batch_size=256, patience=7, arch='mobilenet_v3_large', cache_ram=True):
    
    total_start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        
    # --- Data Setup ---
    if not os.path.exists(data_dir):
        print("❌ Dataset not found.")
        return

    all_paths, all_labels, class_names = find_images_recursively(data_dir)
    
    # Shuffle
    combined = list(zip(all_paths, all_labels))
    import random
    random.shuffle(combined)
    all_paths, all_labels = zip(*combined)
    
    split_idx = int(0.8 * len(all_paths))
    train_ds = CloudDataset(all_paths[:split_idx], all_labels[:split_idx], cache_ram=cache_ram)
    val_ds = CloudDataset(all_paths[split_idx:], all_labels[split_idx:], cache_ram=cache_ram)
    
    # Fast Loader (Main thread only needed if cached)
    workers = 0 if cache_ram else 8
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=workers, pin_memory=True)

    print(f"📂 Data Split: {len(train_ds)} Training, {len(val_ds)} Validation")

    # --- Model Setup ---
    print(f"🏗️  Initializing {arch}...")
    if arch == 'mobilenet_v3_large':
        model = models.mobilenet_v3_large(weights='DEFAULT')
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(class_names))
    elif arch == 'resnet18':
        model = models.resnet18(weights='DEFAULT')
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
    
    model = model.to(device)
    
    # Initialize GPU Augmenter
    gpu_aug = GPUAugment().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2)

    best_acc = 0.0
    best_loss = float('inf')
    epochs_no_improve = 0
    actual_epochs = 0

    print("\n🔥 Starting Training Loop (GPU Accelerated)...")
    
    try:
        for epoch in range(epochs):
            actual_epochs += 1
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0
                total_samples = 0
                start_time = time.time()
                
                loader = train_loader if phase == 'train' else val_loader
                
                for inputs, labels in loader:
                    # 1. Move raw resized tensors to GPU
                    inputs = inputs.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    # 2. Apply Augmentations ON GPU
                    #    (Replaces the slow CPU bottleneck)
                    inputs = gpu_aug(inputs, is_training=(phase=='train'))

                    optimizer.zero_grad()
                    
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()

                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                    total_samples += inputs.size(0)

                epoch_loss = running_loss / total_samples
                epoch_acc = running_corrects.double() / total_samples
                epoch_time = time.time() - start_time
                
                print(f"{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Time: {epoch_time:.1f}s")
                
                if phase == 'train':
                    print(f"   🚀 Speed: {total_samples/epoch_time:.0f} images/sec")

                if phase == 'val':
                    scheduler.step(epoch_loss)
                    if epoch_loss < best_loss:
                        best_loss = epoch_loss
                        best_acc = epoch_acc
                        best_model_wts = copy.deepcopy(model.state_dict())
                        epochs_no_improve = 0
                        print("   ⭐ New Best Model Saved!")
                    else:
                        epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print("🛑 Early stopping triggered!")
                break

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted.")

    print("\n📦 Exporting BEST model to ONNX...")
    model.load_state_dict(best_model_wts)
    model.eval()
    
    # Trace with a dummy input
    dummy = torch.randn(1, 3, 300, 300, device=device)
    # Note: Normalize is part of training loop, not model. 
    # If you want normalization inside ONNX, wrap it. 
    # For now, we export the raw model logic.
    torch.onnx.export(model, dummy, output_model, 
                      input_names=['input'], output_names=['output'],
                      opset_version=18)
    
    with open(output_labels, 'w') as f:
        for name in class_names:
            f.write(f"{name}\n")
    
    print(f"🏆 Done! Best Accuracy: {best_acc:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='dataset')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--arch', type=str, default='mobilenet_v3_large')
    args = parser.parse_args()
    
    # RAM Cache is enabled by default for speed
    train_model(args.data_dir, epochs=args.epochs, batch_size=args.batch_size, arch=args.arch, cache_ram=True)