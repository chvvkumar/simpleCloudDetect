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
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# 1. Memory-Optimized Dataset (Uint8 Caching)
# ---------------------------------------------------------
class CloudDataset(Dataset):
    def __init__(self, file_list, labels, cache_ram=False):
        self.file_list = file_list
        self.labels = labels
        self.cache_ram = cache_ram
        self.cached_images = []
        
        # Resize only. We defer "ToTensor" to keep data as uint8 [0-255]
        self.resize_transform = transforms.Resize((300, 300))

        if self.cache_ram:
            print(f"🧠 Caching {len(file_list)} images to RAM (Uint8 optimized)...")
            for idx, path in enumerate(file_list):
                try:
                    with Image.open(path).convert('RGB') as img:
                        # 1. Resize
                        img = self.resize_transform(img)
                        # 2. Store as Uint8 Tensor [0-255] (Saves 4x RAM)
                        #    Shape: (C, H, W)
                        np_img = np.array(img).transpose(2, 0, 1)
                        tensor = torch.from_numpy(np_img) 
                        self.cached_images.append(tensor)
                except Exception as e:
                    print(f"⚠️ Warning: Could not load {path}: {e}")
                    # Placeholder (Black image)
                    self.cached_images.append(torch.zeros(3, 300, 300, dtype=torch.uint8))
                    
                if (idx + 1) % 5000 == 0:
                    print(f"   Cached {idx + 1} images...")
            print("✅ Cache complete.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        label = self.labels[idx]
        
        if self.cache_ram:
            # Retrieve uint8 tensor -> Convert to float32 [0.0, 1.0]
            # This is extremely fast and happens just before batching
            image = self.cached_images[idx].float() / 255.0
        else:
            # Fallback disk read
            path = self.file_list[idx]
            try:
                with Image.open(path).convert('RGB') as img:
                    img = self.resize_transform(img)
                    image = transforms.functional.to_tensor(img)
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
    if not all_paths:
        print("❌ No images found in dataset directory!")
        return

    # Shuffle
    combined = list(zip(all_paths, all_labels))
    import random
    random.shuffle(combined)