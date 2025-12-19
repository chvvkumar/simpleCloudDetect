# SimpleCloudDetect Modernization: Complete Instruction Set

This document contains the full implementation plan and all necessary code files to migrate `simpleCloudDetect` from a legacy Keras/Teachable Machine workflow to a modern TensorFlow 2.x system.

**Environment Context:**

* **Training:** Windows Subsystem for Linux (WSL) with NVIDIA GPU.

* **Dataset:** Located externally (e.g., mounted Windows drive).

* **Inference:** Docker container on x86 (AMD64) or Raspberry Pi (ARM64).

## Part 1: Master Implementation Plan

**Objective:** Migrate from legacy `.h5` models to native TensorFlow `.keras` models using Transfer Learning (MobileNetV2).

### Phase 1: Training Infrastructure (WSL)

1. **Environment Setup:**

   * Create a `requirements-train.txt` in your WSL environment:

     ```
     tensorflow[and-cuda]
     matplotlib
     numpy
     ```

   * Install: `pip install -r requirements-train.txt`

2. **Dataset Preparation:**

   * Ensure your external dataset follows this structure:

     ```
     /path/to/dataset/
     ├── Clear/
     ├── Mostly Cloudy/
     ├── Overcast/
     ├── Rain/
     ├── Snow/
     └── Wisps/
     ```

3. **Training:**

   * Save the `train.py` code (see Part 2 below) to your working directory.

   * Run the training script pointing to your external dataset:

     ```
     python train.py --data_dir /mnt/d/AllSkyImages/TrainingSet
     ```

   * **Output:** This will generate `model.keras` and `labels.txt`.

### Phase 2: Inference Refactoring & Deployment

1. **Codebase Updates:**

   * Replace `detect.py` with the code in **Part 3**.

   * Replace `Dockerfile` with the code in **Part 4**.

   * Replace `requirements.txt` with the code in **Part 5**.

   * **Delete** `convert.py` and `keras_model.h5` (legacy files are no longer needed).

2. **Build & Run:**

   * Transfer the generated `model.keras` and `labels.txt` to your target machine (if different from the training machine).

   * Build the container:

     ```
     docker build -t simpleclouddetect .
     ```

   * Run the container (mounting the new model files):

     ```
     docker run -d --network=host \
       -v $(pwd)/model.keras:/app/model.keras \
       -v $(pwd)/labels.txt:/app/labels.txt \
       -e IMAGE_URL="http://your-camera-url/image.jpg" \
       -e MQTT_BROKER="192.168.1.x" \
       simpleclouddetect
     ```

## Part 2: Training Script (`train.py`)

Save this file as `train.py`. It is designed to run in WSL and accepts an external dataset path.

```python
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from pathlib import Path

# --- Configuration ---
IMG_SIZE = (224, 224)       # Standard size for MobileNetV2
BATCH_SIZE = 32
EPOCHS_INITIAL = 10         # Epochs for frozen base training
EPOCHS_FINE = 10            # Epochs for fine-tuning
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'model.keras'
LABELS_SAVE_PATH = 'labels.txt'

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cloud Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the dataset directory containing class folders')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # 1. GPU Check
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s). Training will be accelerated.")
    else:
        print("⚠️ No GPU found. Training will use CPU.")

    # 2. Load Data
    if not data_dir.exists():
        print(f"❌ Error: Dataset directory '{data_dir}' not found.")
        return

    print(f"Loading dataset from: {data_dir}")
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
    except ValueError as e:
        print(f"❌ Error loading dataset: {e}")
        print("Ensure the directory contains subfolders for each class.")
        return

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # 3. Optimize Data Loading
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # 5. Create Model (MobileNetV2 Transfer Learning)
    # Note: MobileNetV2 specific preprocessing is included in the model pipeline
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)  # Scales inputs appropriately
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # 6. Initial Training
    print("--- Starting Phase 1: Training Head ---")
    history = model.fit(
        train_ds,
        epochs=EPOCHS_INITIAL,
        validation_data=val_ds
    )

    # 7. Fine-Tuning
    print("--- Starting Phase 2: Fine-Tuning Base ---")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE/10),
        metrics=['accuracy']
    )
    
    total_epochs = EPOCHS_INITIAL + EPOCHS_FINE
    history_fine = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds
    )

    # 8. Save Artifacts
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    print(f"Saving labels to {LABELS_SAVE_PATH}...")
    with open(LABELS_SAVE_PATH, 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    print("✅ Training complete.")

if __name__ == "__main__":
    main()
```

## Part 3: Modern Detection Script (`detect.py`)

Replace the existing `detect.py` with this version. It removes legacy Keras imports and handles the new model format properly.

```python
#!/usr/bin/env python3

import logging
import os
import socket
import time
import json
import gc
import io
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import paho.mqtt.client as mqtt
import requests
from PIL import Image, ImageOps
import tensorflow as tf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class Config:
    """Configuration class to hold all environment variables"""
    image_url: str
    model_path: str
    label_path: str
    broker: str
    port: int
    topic: str
    detect_interval: int
    mqtt_username: Optional[str]
    mqtt_password: Optional[str]
    mqtt_discovery_mode: str
    mqtt_discovery_prefix: str
    device_name: str
    device_id: Optional[str]

    @classmethod
    def from_env(cls) -> 'Config':
        """Create Config from environment variables with validation"""
        discovery_mode = os.getenv('MQTT_DISCOVERY_MODE', 'legacy').lower()
        
        required_vars = ['IMAGE_URL', 'MQTT_BROKER', 'DETECT_INTERVAL']
        
        if discovery_mode == 'legacy':
            required_vars.append('MQTT_TOPIC')
        elif discovery_mode == 'homeassistant':
            if not os.getenv('DEVICE_ID'):
                raise ValueError("DEVICE_ID is required when MQTT_DISCOVERY_MODE is 'homeassistant'")
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return cls(
            image_url=os.environ['IMAGE_URL'],
            model_path=os.getenv('MODEL_PATH', 'model.keras'),
            label_path=os.getenv('LABEL_PATH', 'labels.txt'),
            broker=os.environ['MQTT_BROKER'],
            port=int(os.getenv('MQTT_PORT', '1883')),
            topic=os.getenv('MQTT_TOPIC', ''),
            detect_interval=int(os.environ['DETECT_INTERVAL']),
            mqtt_username=os.getenv('MQTT_USERNAME'),
            mqtt_password=os.getenv('MQTT_PASSWORD'),
            mqtt_discovery_mode=discovery_mode,
            mqtt_discovery_prefix=os.getenv('MQTT_DISCOVERY_PREFIX', 'homeassistant'),
            device_name=os.getenv('DEVICE_NAME', 'Cloud Detector'),
            device_id=os.getenv('DEVICE_ID')
        )

class HADiscoveryManager:
    """Manages Home Assistant MQTT Discovery"""
    def __init__(self, config: Config, mqtt_client):
        self.config = config
        self.mqtt_client = mqtt_client
        self.device_id = config.device_id
        self.discovery_prefix = config.mqtt_discovery_prefix
        self.availability_topic = f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/availability"
        
    def get_device_info(self) -> dict:
        return {
            "identifiers": [f"clouddetect_{self.device_id}"],
            "name": self.config.device_name,
            "manufacturer": "chvvkumar",
            "model": "ML Cloud Detection (Modern)",
            "sw_version": "2.0"
        }
    
    def publish_discovery_configs(self):
        device_info = self.get_device_info()
        
        # Cloud Status Sensor
        status_config = {
            "name": "Status",
            "unique_id": f"clouddetect_{self.device_id}_status",
            "state_topic": f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/status/state",
            "availability_topic": self.availability_topic,
            "icon": "mdi:weather-cloudy",
            "device": device_info
        }
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/status/config",
            json.dumps(status_config),
            retain=True
        )
        
        # Confidence Sensor
        confidence_config = {
            "name": "Confidence",
            "unique_id": f"clouddetect_{self.device_id}_confidence",
            "state_topic": f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/confidence/state",
            "availability_topic": self.availability_topic,
            "unit_of_measurement": "%",
            "icon": "mdi:percent",
            "device": device_info
        }
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/confidence/config",
            json.dumps(confidence_config),
            retain=True
        )
        
        # Detection Time Sensor
        time_config = {
            "name": "Detection Time",
            "unique_id": f"clouddetect_{self.device_id}_detection_time",
            "state_topic": f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/detection_time/state",
            "availability_topic": self.availability_topic,
            "unit_of_measurement": "s",
            "device_class": "duration",
            "icon": "mdi:timer",
            "device": device_info
        }
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/detection_time/config",
            json.dumps(time_config),
            retain=True
        )
        
        self.mqtt_client.publish(self.availability_topic, "online", retain=True)

    def publish_states(self, result: dict):
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/status/state",
            result["class_name"]
        )
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/confidence/state",
            str(result["confidence_score"])
        )
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/detection_time/state",
            str(result["Detection Time (Seconds)"])
        )

class CloudDetector:
    """Main class for cloud detection operations"""
    def __init__(self, config: Config, mqtt_client=None):
        self.config = config
        self.model = self._load_model()
        self.class_names = self._load_class_names()
        self.session = requests.Session()
        self.mqtt_client = mqtt_client if mqtt_client is not None else self._setup_mqtt()
        self.ha_discovery = None
        
        if self.config.mqtt_discovery_mode == 'homeassistant':
            self.ha_discovery = HADiscoveryManager(self.config, self.mqtt_client)
            self.ha_discovery.publish_discovery_configs()
        
    def _load_model(self):
        """Load and return the TensorFlow model"""
        try:
            logger.info(f"Loading model from {self.config.model_path}...")
            return tf.keras.models.load_model(self.config.model_path, compile=False)
        except Exception as e:
            logger.error(f"Failed to load model from {self.config.model_path}: {e}")
            raise

    def _load_class_names(self):
        try:
            with open(self.config.label_path, "r") as f:
                # Strip newlines and empty strings
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception as e:
            logger.error(f"Failed to load labels from {self.config.label_path}: {e}")
            raise

    def _setup_mqtt(self):
        if not self.config.broker or self.config.broker.lower() in ['none', 'null', '']:
            return None
        client = mqtt.Client()
        if self.config.mqtt_username and self.config.mqtt_password:
            client.username_pw_set(self.config.mqtt_username, self.config.mqtt_password)
        if self.config.mqtt_discovery_mode == 'homeassistant':
            availability_topic = f"{self.config.mqtt_discovery_prefix}/sensor/clouddetect_{self.config.device_id}/availability"
            client.will_set(availability_topic, "offline", retain=True)
        try:
            client.connect(self.config.broker, self.config.port)
            client.loop_start()
            return client
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def _load_image(self, image_url: str, max_retries: int = 3) -> Image.Image:
        for attempt in range(max_retries):
            try:
                if image_url.startswith("file://"):
                    parsed = urllib.parse.urlparse(image_url)
                    file_path = Path(parsed.path.lstrip('/'))
                    if not file_path.exists():
                        raise FileNotFoundError(f"Image file not found: {file_path}")
                    with open(file_path, 'rb') as f:
                        return Image.open(f).convert("RGB")
                else:
                    with self.session.get(image_url, timeout=(5, 10), stream=True) as response:
                        response.raise_for_status()
                        return Image.open(io.BytesIO(response.content)).convert("RGB")
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to load image: {e}")
                    raise
                time.sleep(1)

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize to 224x224 (MobileNetV2 standard)
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        
        # NOTE: MobileNetV2 with transfer learning typically expects raw [0, 255] 
        # or specific normalization. The training script uses the model's built-in 
        # preprocessing layer, which handles normalization internally.
        # So we just expand dims here.
        return np.expand_dims(image_array, axis=0)

    def detect(self, return_image: bool = False) -> dict:
        start_time = time.time()
        try:
            image = self._load_image(self.config.image_url)
            preprocessed_image = self._preprocess_image(image)
            
            prediction = self.model.predict(preprocessed_image, verbose=0)
            index = np.argmax(prediction)
            
            if index < len(self.class_names):
                class_name = self.class_names[index]
            else:
                class_name = "Unknown"
            
            confidence_score = float(prediction[0][index])
            elapsed_time = time.time() - start_time
            
            result = {
                "class_name": class_name,
                "confidence_score": round(confidence_score * 100, 2),
                "Detection Time (Seconds)": round(elapsed_time, 2)
            }
            
            if return_image:
                result['image'] = image.copy()
            
            # Explicit cleanup for low-memory devices
            del image
            del preprocessed_image
            gc.collect()
            
            logger.info(f"Detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

    def publish_result(self, result: dict):
        if not self.mqtt_client:
            return
        try:
            if self.config.mqtt_discovery_mode == 'homeassistant':
                self.ha_discovery.publish_states(result)
            else:
                json_result = json.dumps(result)
                self.mqtt_client.publish(self.config.topic, json_result)
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")

    def run_detection_loop(self):
        logger.info("Starting detection loop (Modern TF)...")
        while True:
            try:
                result = self.detect()
                self.publish_result(result)
                time.sleep(self.config.detect_interval)
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(5)

def main():
    socket.setdefaulttimeout(30)
    try:
        config = Config.from_env()
        detector = CloudDetector(config)
        detector.run_detection_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
```

## Part 4: Dockerfile

This is the multi-arch `Dockerfile` optimized for the new workflow.

```dockerfile
# Use a slim python image (Compatible with both AMD64 and ARM64)
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# libgl1-mesa-glx and libglib2.0-0 are often needed for cv2 or pillow features
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY detect.py .
COPY alpaca_safety_monitor.py .
COPY start_services.sh .
COPY templates/ templates/
# Note: users should mount model.keras and labels.txt via volumes, 
# but you can COPY defaults here if you have them.

# Make start script executable
RUN chmod +x start_services.sh

# Environment variables
ENV PYTHONUNBUFFERED=1

# Run the start script
CMD ["./start_services.sh"]
```

## Part 5: Requirements (`requirements.txt`)

These are the dependencies for the **Docker Container** (Runtime).

```text
tensorflow-cpu==2.18.0
numpy
pillow
paho-mqtt
requests
flask
flask-cors
waitress
typing_extensions
```