#!/usr/bin/env python3

import logging
import os
import time
import json
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import paho.mqtt.client as mqtt
import requests
from PIL import Image, ImageOps
from keras.models import load_model

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

    @classmethod
    def from_env(cls) -> 'Config':
        """Create Config from environment variables with validation"""
        required_vars = ['IMAGE_URL', 'MQTT_BROKER', 'MQTT_TOPIC', 'DETECT_INTERVAL']
        missing = [var for var in required_vars if not os.getenv(var)]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
        
        return cls(
            image_url=os.environ['IMAGE_URL'],
            model_path=os.getenv('MODEL_PATH', 'keras_model.h5'),
            label_path=os.getenv('LABEL_PATH', 'labels.txt'),
            broker=os.environ['MQTT_BROKER'],
            port=int(os.getenv('MQTT_PORT', '1883')),
            topic=os.environ['MQTT_TOPIC'],
            detect_interval=int(os.environ['DETECT_INTERVAL']),
            mqtt_username=os.getenv('MQTT_USERNAME'),
            mqtt_password=os.getenv('MQTT_PASSWORD')
        )

class CloudDetector:
    """Main class for cloud detection operations"""
    def __init__(self, config: Config):
        self.config = config
        self.model = self._load_model()
        self.class_names = self._load_class_names()
        self.mqtt_client = self._setup_mqtt()
        
    def _load_model(self):
        """Load and return the Keras model"""
        try:
            return load_model(self.config.model_path, compile=False)
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
        client = mqtt.Client()
        if self.config.mqtt_username and self.config.mqtt_password:
            client.username_pw_set(self.config.mqtt_username, self.config.mqtt_password)
        
        try:
            client.connect(self.config.broker, self.config.port)
            logger.info(f"Connected to MQTT broker at {self.config.broker}:{self.config.port}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            raise

    def _load_image(self, image_url: str, max_retries: int = 3) -> Image.Image:
        """Load and return image from URL or file with retry logic"""
        for attempt in range(max_retries):
            try:
                if image_url.startswith("file://"):
                    # Handle file URLs
                    parsed = urllib.parse.urlparse(image_url)
                    file_path = Path(parsed.path.lstrip('/'))  # Remove leading slashes
                    if not file_path.exists():
                        raise FileNotFoundError(f"Image file not found: {file_path}")
                    return Image.open(file_path).convert("RGB")
                else:
                    # Handle HTTP URLs
                    response = requests.get(image_url, timeout=10, stream=True)
                    response.raise_for_status()
                    return Image.open(response.raw).convert("RGB")
            except (requests.RequestException, IOError) as e:
                if attempt == max_retries - 1:  # Last attempt
                    logger.error(f"Failed to load image from {image_url} after {max_retries} attempts: {e}")
                    raise
                logger.warning(f"Attempt {attempt + 1}/{max_retries} failed: {e}")
                time.sleep(1)  # Wait before retrying

    def _preprocess_image(self, image: Image.Image) -> np.ndarray:
        """Preprocess image for model input"""
        # Resize and normalize image
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        image_array = np.asarray(image)
        normalized_array = (image_array.astype(np.float32) / 127.5) - 1
        return np.expand_dims(normalized_array, axis=0)

    def detect(self) -> dict:
        """Perform cloud detection on an image"""
        start_time = time.time()
        
        try:
            # Load and preprocess image
            image = self._load_image(self.config.image_url)
            preprocessed_image = self._preprocess_image(image)
            
            # Make prediction
            prediction = self.model.predict(preprocessed_image, verbose=0)
            index = np.argmax(prediction)
            class_name = self.class_names[index].strip()[2:]  # Remove index and newline
            confidence_score = float(prediction[0][index])
            
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
    try:
        config = Config.from_env()
        detector = CloudDetector(config)
        detector.run_detection_loop()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()
