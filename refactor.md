# Modernization Plan: Keras/TF to PyTorch/ONNX

This plan converts the `simpleCloudDetect` stack to use **PyTorch** for GPU-accelerated training on Windows (WSL) and **ONNX Runtime** for lightweight CPU inference on Raspberry Pi.

---

## Phase 1: Environment Setup (WSL)

**Objective:** Create a reproducible Python environment on WSL supporting NVIDIA GPUs and link your existing dataset.

**Script:** `setup_env.sh`

```bash
#!/bin/bash

# 1. Update System & Install Python Dependencies
echo "Updating system..."
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv

# 2. Create Project Directory
PROJECT_DIR=~/simpleCloudDetect_Modern
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 3. Create Virtual Environment
echo "Creating virtual environment..."
python3 -m venv venv

# 4. Activate Environment
source venv/bin/activate

# 5. Install PyTorch with CUDA Support (for NVIDIA GPU)
# Note: Assumes CUDA drivers are installed on Windows host
echo "Installing PyTorch (CUDA)..."
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# 6. Install Other Utilities
echo "Installing utilities..."
pip install onnx onnxruntime pillow numpy requests

# 7. Create Symlink to Dataset (Optional but convenient)
# This lets you refer to 'dataset' instead of the full path
if [ -d "/mnt/f/MLClouds_incoming/resized/" ]; then
    echo "Linking dataset..."
    ln -s /mnt/f/MLClouds_incoming/resized/ dataset
else
    echo "Warning: Dataset path /mnt/f/MLClouds_incoming/resized/ not found."
fi

echo "Setup complete! Activate via: source ~/simpleCloudDetect_Modern/venv/bin/activate"
```

---

## Phase 2: Model Training (Windows/WSL)

**Objective:** Use a custom script to recursively load images from your specific folder structure, fine-tune MobileNetV3, and export to ONNX.

**Instructions:**
1. Ensure your dataset is at `/mnt/f/MLClouds_incoming/resized/`.
2. The script expects the structure: `resized/ClassName/DateFolder/image.jpg` (or any depth of subfolders).

**Script:** `train_model.py`

```python
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
```

---

## Phase 3: Runtime Modernization (Raspberry Pi)

**Objective:** Update `detect.py` to use `onnxruntime` instead of Keras.

**Script:** `detect.py`

```python
import logging
import os
import time
import json
import io
import requests
import numpy as np
import paho.mqtt.client as mqtt
import onnxruntime as ort
from PIL import Image, ImageOps
from dataclasses import dataclass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Config:
    image_url: str
    model_path: str
    label_path: str
    broker: str
    port: int
    topic: str
    detect_interval: int
    mqtt_username: str = None
    mqtt_password: str = None
    mqtt_discovery_mode: str = 'legacy'
    mqtt_discovery_prefix: str = 'homeassistant'
    device_name: str = 'Cloud Detector'
    device_id: str = None

    @classmethod
    def from_env(cls):
        return cls(
            image_url=os.environ['IMAGE_URL'],
            model_path=os.getenv('MODEL_PATH', 'model.onnx'),
            label_path=os.getenv('LABEL_PATH', 'labels.txt'),
            broker=os.environ.get('MQTT_BROKER'),
            port=int(os.getenv('MQTT_PORT', '1883')),
            topic=os.getenv('MQTT_TOPIC', 'Astro/CloudDetect'),
            detect_interval=int(os.environ.get('DETECT_INTERVAL', '60')),
            mqtt_username=os.getenv('MQTT_USERNAME'),
            mqtt_password=os.getenv('MQTT_PASSWORD'),
            mqtt_discovery_mode=os.getenv('MQTT_DISCOVERY_MODE', 'legacy'),
            mqtt_discovery_prefix=os.getenv('MQTT_DISCOVERY_PREFIX', 'homeassistant'),
            device_name=os.getenv('DEVICE_NAME', 'Cloud Detector'),
            device_id=os.getenv('DEVICE_ID')
        )

class CloudDetector:
    def __init__(self, config):
        self.config = config
        self.labels = self._load_labels()
        # Use CPU Provider for Pi
        self.ort_session = ort.InferenceSession(self.config.model_path, providers=['CPUExecutionProvider'])
        self.mqtt = self._setup_mqtt()

    def _load_labels(self):
        with open(self.config.label_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _setup_mqtt(self):
        if not self.config.broker: return None
        client = mqtt.Client()
        if self.config.mqtt_username and self.config.mqtt_password:
            client.username_pw_set(self.config.mqtt_username, self.config.mqtt_password)
        try:
            client.connect(self.config.broker, self.config.port)
            client.loop_start()
            logger.info(f"Connected to MQTT: {self.config.broker}")
            return client
        except Exception as e:
            logger.error(f"MQTT Error: {e}")
            return None

    def _preprocess(self, image):
        # Resize
        img = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        # Normalize (ImageNet stats)
        img_data = np.array(img).astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_data = (img_data - mean) / std
        # Transpose to (Batch, Channel, Height, Width)
        img_data = img_data.transpose(2, 0, 1)
        return np.expand_dims(img_data, axis=0)

    def detect(self):
        try:
            # Fetch
            if self.config.image_url.startswith('file://'):
                path = self.config.image_url.replace('file://', '')
                img = Image.open(path).convert('RGB')
            else:
                resp = requests.get(self.config.image_url, stream=True, timeout=10)
                img = Image.open(io.BytesIO(resp.content)).convert('RGB')
            
            # Predict
            input_tensor = self._preprocess(img)
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: input_tensor})
            
            # Softmax
            logits = outputs[0][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            idx = np.argmax(probs)
            
            result = {
                "class_name": self.labels[idx],
                "confidence_score": round(float(probs[idx]) * 100, 2),
                "timestamp": time.time()
            }
            logger.info(f"Result: {result}")
            
            # Publish
            if self.mqtt:
                self.mqtt.publish(self.config.topic, json.dumps(result))
                
            return result

        except Exception as e:
            logger.error(f"Error: {e}")
            return None

    def run(self):
        logger.info("Starting loop...")
        while True:
            self.detect()
            time.sleep(self.config.detect_interval)

if __name__ == "__main__":
    detector = CloudDetector(Config.from_env())
    detector.run()
```

---

## Phase 4: Packaging (Docker)

**Objective:** Reduce container size by removing TensorFlow.

**File:** `requirements.txt`
```text
onnxruntime==1.17.1
pillow==10.2.0
numpy==1.26.4
paho-mqtt==1.6.1
requests==2.31.0
flask==3.0.0
flask-cors==4.0.0
waitress==3.0.0
```

**File:** `Dockerfile`
```dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy App
COPY . .

# Run
CMD ["python", "detect.py"]
```