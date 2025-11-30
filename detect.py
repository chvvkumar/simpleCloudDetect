import os
import time
import schedule
import logging
import json
import socket
import urllib.request
import numpy as np
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# --- CONFIGURATION ---
# (Ensure these match your environment/docker-compose settings)
MQTT_BROKER = os.getenv('MQTT_BROKER', '192.168.1.250')
MQTT_PORT = int(os.getenv('MQTT_PORT', 1883))
MQTT_TOPIC = "Astro/SimpleCloudDetect"
IMAGE_URL = os.getenv('IMAGE_URL', 'https://allsky.challa.co:1982/current/resized/image.jpg') # Replace with your camera URL

# Set global timeout for all network operations (Critical Fix for Hangs)
socket.setdefaulttimeout(30)

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- LOAD LABELS & MODEL ---
# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

try:
    with open("labels.txt", "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    # Load the model
    model = load_model("keras_model.h5", compile=False)
    logger.info("Model and labels loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model or labels: {e}")
    exit(1)

# --- MQTT SETUP ---
client = mqtt.Client()

def on_connect(client, userdata, flags, rc):
    if rc == 0:
        logger.info("Connected to MQTT Broker")
    else:
        logger.error(f"Failed to connect to MQTT Broker, return code {rc}")

client.on_connect = on_connect

try:
    client.connect(MQTT_BROKER, MQTT_PORT, 60)
    # Critical Fix: Run a background loop to handle reconnections and heartbeats
    client.loop_start() 
except Exception as e:
    logger.error(f"Could not connect to MQTT Broker: {e}")

# --- DETECTION JOB ---
def job():
    start_time = time.time()
    img_path = "latest.jpg"
    
    try:
        # 1. Download Image (Will now timeout if stuck)
        urllib.request.urlretrieve(IMAGE_URL, img_path)
        
        # 2. Preprocess Image
        # Open and resize to 224x224 (standard for Teachable Machine models)
        image_file = Image.open(img_path).convert("RGB")
        image_file = image_file.resize((224, 224))
        
        img_array = np.asarray(image_file)
        
        # Normalize the image array
        normalized_image_array = (img_array.astype(np.float32) / 127.5) - 1
        
        # Load the image into the array
        data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
        data[0] = normalized_image_array
        
        # 3. Predict
        prediction = model.predict(data, verbose=0)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]

        # 4. Prepare Payload
        end_time = time.time()
        duration = round(end_time - start_time, 2)
        
        # Clean class name (remove index numbers if present, e.g. "0 Clear" -> "Clear")
        clean_class_name = class_name[2:].strip() if len(class_name) > 2 and class_name[0].isdigit() else class_name

        payload = {
            "class_name": clean_class_name,
            "confidence_score": round(float(confidence_score) * 100, 2),
            "Detection Time (Seconds)": duration
        }
        
        logger.info(f"Detection: {payload}")
        
        # 5. Publish
        client.publish(MQTT_TOPIC, json.dumps(payload))
        logger.info(f"Published to {MQTT_TOPIC}: {json.dumps(payload)}")

    except socket.timeout:
        logger.error("Timeout: Camera image download took too long. Skipping this cycle.")
    except Exception as e:
        logger.error(f"Error during detection cycle: {e}")
    finally:
        # cleanup image to save space/ensure fresh download next time
        if os.path.exists(img_path):
            os.remove(img_path)

# --- MAIN LOOP ---
# Run immediately on startup
job()

# Schedule every 60 seconds
schedule.every(60).seconds.do(job)

logger.info("Service started. Running detection every 60 seconds...")

while True:
    schedule.run_pending()
    time.sleep(1)