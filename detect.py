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
import onnxruntime as ort

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
    # HA Discovery settings
    mqtt_discovery_mode: str
    mqtt_discovery_prefix: str
    device_name: str
    device_id: Optional[str]

    @classmethod
    def from_env(cls) -> 'Config':
        """Create Config from environment variables with validation"""
        discovery_mode = os.getenv('MQTT_DISCOVERY_MODE', 'legacy').lower()
        
        # Base required vars for all modes
        required_vars = ['IMAGE_URL', 'MQTT_BROKER', 'DETECT_INTERVAL']
        
        # Legacy mode requires MQTT_TOPIC
        if discovery_mode == 'legacy':
            required_vars.append('MQTT_TOPIC')
        # HA discovery mode requires DEVICE_ID
        elif discovery_mode == 'homeassistant':
            if not os.getenv('DEVICE_ID'):
                raise ValueError("DEVICE_ID is required when MQTT_DISCOVERY_MODE is 'homeassistant'")
        
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return cls(
            image_url=os.environ['IMAGE_URL'],
            model_path=os.getenv('MODEL_PATH', 'model.onnx'),
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
        """Get device information dict for HA discovery"""
        return {
            "identifiers": [f"clouddetect_{self.device_id}"],
            "name": self.config.device_name,
            "manufacturer": "chvvkumar",
            "model": "ML Cloud Detection",
            "sw_version": "1.0"
        }
    
    def publish_discovery_configs(self):
        """Publish discovery configuration for all sensors"""
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
        
        # Confidence Score Sensor
        confidence_config = {
            "name": "Confidence",
            "unique_id": f"clouddetect_{self.device_id}_confidence",
            "state_topic": f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}_confidence/state",
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
            "state_topic": f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}_detection_time/state",
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
        
        # Publish availability as online
        self.mqtt_client.publish(self.availability_topic, "online", retain=True)
        logger.info("Published HA discovery configurations")
    
    def publish_states(self, result: dict):
        """Publish individual sensor states for HA discovery mode"""
        # Publish cloud status
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/status/state",
            result["class_name"]
        )
        
        # Publish confidence score
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/confidence/state",
            str(result["confidence_score"])
        )
        
        # Publish detection time
        self.mqtt_client.publish(
            f"{self.discovery_prefix}/sensor/clouddetect_{self.device_id}/detection_time/state",
            str(result["Detection Time (Seconds)"])
        )
        
        logger.info(f"Published HA states: {result['class_name']}, {result['confidence_score']}%, {result['Detection Time (Seconds)']}s")


class CloudDetector:
    """Main class for cloud detection operations"""
    def __init__(self, config: Config, mqtt_client=None):
        self.config = config
        self.model = self._load_model()
        self.class_names = self._load_class_names()
        self.session = requests.Session()  # Reuse HTTP connections
        self.mqtt_client = mqtt_client if mqtt_client is not None else self._setup_mqtt()
        self.ha_discovery = None
        
        # Initialize HA discovery if enabled
        if self.config.mqtt_discovery_mode == 'homeassistant':
            self.ha_discovery = HADiscoveryManager(self.config, self.mqtt_client)
            self.ha_discovery.publish_discovery_configs()
        
    def _load_model(self):
        """Load and return the ONNX model session"""
        try:
            # Use CPU Provider for Raspberry Pi
            return ort.InferenceSession(self.config.model_path, providers=['CPUExecutionProvider'])
        except Exception as e:
            logger.error(f"Failed to load model from {self.config.model_path}: {e}")
            raise

    def _load_class_names(self):
        """Load and return class names from the labels file"""
        try:
            with open(self.config.label_path, "r") as f:
                return f.readlines()
        except Exception as e:
            logger.error(f"Failed to load labels from {self.config.label_path}: {e}")
            raise

    def _setup_mqtt(self):
        """Setup and return MQTT client"""
        # Return None if broker is not configured
        if not self.config.broker or self.config.broker.lower() in ['none', 'null', '']:
            logger.warning("MQTT broker not configured, MQTT publishing disabled")
            return None
            
        client = mqtt.Client()
        if self.config.mqtt_username and self.config.mqtt_password:
            client.username_pw_set(self.config.mqtt_username, self.config.mqtt_password)
        
        # Set Last Will and Testament for HA discovery mode
        if self.config.mqtt_discovery_mode == 'homeassistant':
            availability_topic = f"{self.config.mqtt_discovery_prefix}/sensor/clouddetect_{self.config.device_id}/availability"
            client.will_set(availability_topic, "offline", retain=True)
        
        try:
            client.connect(self.config.broker, self.config.port)
            # Start background network loop to handle keepalive pings and reconnections
            client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.config.broker}:{self.config.port}")
            logger.info(f"MQTT Discovery Mode: {self.config.mqtt_discovery_mode}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def _load_image(self, image_url: str, max_retries: int = 3) -> Image.Image:
        """Load and return image from URL or file with retry logic and strict timeouts"""
        for attempt in range(max_retries):
            try:
                if image_url.startswith("file://"):
                    # Handle file URLs
                    parsed = urllib.parse.urlparse(image_url)
                    file_path = Path(parsed.path.lstrip('/'))  # Remove leading slashes
                    if not file_path.exists():
                        raise FileNotFoundError(f"Image file not found: {file_path}")
                    with open(file_path, 'rb') as f:
                        return Image.open(f).convert("RGB")
                else:
                    # FIX: Strict timeout (5s connect, 10s read) and close socket immediately
                    with self.session.get(image_url, timeout=(5, 10), stream=True) as response:
                        response.raise_for_status()
                        # Load into memory buffer to allow socket to close
                        return Image.open(io.BytesIO(response.content)).convert("RGB")
            except (requests.RequestException, IOError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Failed to load image from {image_url} after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)  # Small backoff between retries

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        # --- CHANGED: Use simple resize to match training script (300x300) ---
        # Match the PyTorch transforms.Resize((300, 300)) logic
        image = image.resize((300, 300), Image.Resampling.LANCZOS)
        
        # Normalize (ImageNet stats)
        img_data = np.array(image).astype(np.float32) / 255.0
        
        # --- FIX: Explicitly use float32 for mean/std to prevent auto-promoting to double ---
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        img_data = (img_data - mean) / std
        
        # Transpose to (Batch, Channel, Height, Width)
        img_data = img_data.transpose(2, 0, 1)
        return np.expand_dims(img_data, axis=0)

    def detect(self) -> dict:
        """Perform cloud detection on an image"""
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self._load_image(self.config.image_url)
            preprocessed_image = self._preprocess_image(image)
            
            # Make prediction with ONNX
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: preprocessed_image})
            
            # Softmax
            logits = outputs[0][0]
            probs = np.exp(logits) / np.sum(np.exp(logits))
            index = np.argmax(probs)
            
            # FIX: Robust label parsing - handles "0 Clear" -> "Clear" format
            if index < len(self.class_names):
                raw_label = self.class_names[index].strip()
                # Check if label starts with digit followed by space (e.g., "0 Clear")
                if " " in raw_label and raw_label.split(" ", 1)[0].isdigit():
                    class_name = raw_label.split(" ", 1)[1]
                else:
                    class_name = raw_label
            else:
                class_name = "Unknown"
            
            confidence_score = float(probs[index])
            
            # FIX: Force garbage collection to prevent memory fragmentation on Pi
            del image
            del preprocessed_image
            gc.collect()
            
            elapsed_time = time.time() - start_time
            
            result = {
                "class_name": class_name,
                "confidence_score": round(confidence_score * 100, 2),
                "Detection Time (Seconds)": round(elapsed_time, 2)
            }
            
            logger.info(f"Detection: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Detection failed: {e}")
            raise

    def publish_result(self, result: dict):
        """Publish detection result to MQTT"""
        try:
            if self.config.mqtt_discovery_mode == 'homeassistant':
                # Use HA discovery publishing
                self.ha_discovery.publish_states(result)
            else:
                # Use legacy single-topic publishing
                json_result = json.dumps(result)
                self.mqtt_client.publish(self.config.topic, json_result)
                logger.info(f"Published to {self.config.topic}: {json_result}")
        except Exception as e:
            logger.error(f"Failed to publish result: {e}")
            raise

    def run_detection_loop(self):
        """Main detection loop"""
        logger.info("Starting detection loop...")
        while True:
            try:
                result = self.detect()
                self.publish_result(result)
                time.sleep(self.config.detect_interval)
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                time.sleep(5)  # Wait before retrying

def main():
    """Main entry point"""
    # Set global socket timeout to prevent network operations from hanging indefinitely
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
