import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from PIL import Image

# Custom Dataset to handle recursive folder structures
# (e.g. Class/Date/Image.jpg)
class RecursiveImageFolder(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.images = self._find_images()

    def _find_images(self):
        image_list = []
        # Walk through each class folder
        for cls_name in self.classes:
            class_dir = os.path.join(self.root_dir, cls_name)
            class_idx = self.class_to_idx[cls_name]
            
            # Recursively walk through all subdirectories (dates, etc.)
            for root, _, files in os.walk(class_dir):
                for file in files:
                    if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                        path = os.path.join(root, file)
                        image_list.append((path, class_idx))
        return image_list

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        path, label = self.images[idx]
        try:
            image = Image.open(path).convert('RGB')
        except Exception as e:
            print(f"Skipping corrupt image: {path}")
            return self.__getitem__((idx + 1) % len(self))
            
        if self.transform:
            image = self.transform(image)
        return image, label

def train_model(data_dir, output_model='model.onnx', output_labels='labels.txt', epochs=20):
    # Detect GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Training on: {device}")

    # Modern Data Augmentation
    data_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load Data recursively
    if not os.path.exists(data_dir):
        print(f"❌ Error: '{data_dir}' not found.")
        return

    print(f"🔍 Scanning for images in {data_dir}...")
    dataset = RecursiveImageFolder(data_dir, transform=data_transforms)
    
    if len(dataset) == 0:
        print("❌ No images found! Check your path and structure.")
        return

    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    print(f"📂 Found {len(dataset)} images in {len(dataset.classes)} classes: {dataset.classes}")

    # Save clean labels
    with open(output_labels, 'w') as f:
        for name in dataset.classes:
            f.write(f"{name}\n")

    # Load MobileNetV3 (Small)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    
    # Modify classifier
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, len(dataset.classes))
    model = model.to(device)

    # Training Loop
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("🔥 Starting training...")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        acc = 100 * correct / total
        print(f"   Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(dataloader):.4f} | Accuracy: {acc:.2f}%")

    # Export to ONNX
    print("📦 Exporting to ONNX...")
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224, device=device)
    
    torch.onnx.export(model, dummy_input, output_model, 
                      input_names=['input'], output_names=['output'],
                      opset_version=12)
    
    print(f"✅ Success! Saved '{output_model}' and '{output_labels}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Default to the WSL path provided
    parser.add_argument('--data_dir', type=str, default='/mnt/f/MLClouds_incoming/resized/', help='Path to dataset folder')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    args = parser.parse_args()
    
    train_model(args.data_dir, epochs=args.epochs)
