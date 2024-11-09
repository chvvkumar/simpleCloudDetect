#!/home/pi/git/simpleCloudDetect/env/bin/python

from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np
import json
import paho.mqtt.client as mqtt
import requests
from io import BytesIO
import os
import time

# Define parameters
image_url = os.environ['IMAGE_URL']
broker = os.environ['MQTT_BROKER']
port = int(os.getenv("MQTT_PORT"))
topic = os.environ['MQTT_TOPIC']
detect_interval =  os.environ['DETECT_INTERVAL']

# Load the model and class names
model = load_model("keras_model.h5", compile=False) # Load the model
class_names = open("labels.txt", "r").readlines() # Load the class names

# Clear the console
os.system('cls' if os.name == 'nt' else 'clear')

# Connect to the MQTT broker
client = mqtt.Client()
client.connect(broker, port)
print("Connected to MQTT broker at:", broker, "on port:", port, "with topic:", topic)

# Function to detect an object in an image and publish the result to an MQTT topic
def detect(image_url, topic, model, class_names):
    # Get the image from the URL
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content)).convert("RGB")

    # Start the timer
    start_time = time.time()

    # resizing the image to be at least 224x224 and then cropping from the center
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

    # turn the image into a numpy array
    image_array = np.asarray(image)

    # Normalize the image
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

    # Create the array of the right shape to feed into the keras model
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32) # 3 is the number of channels (RGB) in the image
    data[0] = normalized_image_array # Load the image into the array

    # Predicts the model
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index].strip() # Remove the newline character from the class name
    confidence_score = prediction[0][index] # Get the confidence score of the prediction

    # End the timer and calculate the elapsed time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Print the prediction, confidence score, and elapsed time 
    print("Class:", class_name[2:], "Confidence Score:", confidence_score, "Elapsed Time:", round(elapsed_time, 2))

    # Create a dictionary with the class name and confidence score
    data = {
        "class_name": class_name[2:], # Remove the index number from the class name
        "confidence_score": round(float(confidence_score) * 100, 2), # Convert the confidence score to a percentage
        "Detection Time (Seconds)": round(elapsed_time, 2) # Round the elapsed time to 2 decimal places
    }

    # Convert dictionary to JSON object
    json_object = json.dumps(data) # Convert the dictionary to a JSON object

    # Return the JSON object
    return json_object

# Function to publish the JSON object to an MQTT topic
def publish(json_object):
    # Publish the JSON object to the MQTT topic
    client.publish(topic, json_object)
    print("Published data to MQTT topic:", topic, "Data:", json_object)


# Main function
if __name__ == "__main__":
    while True:
        # Call the function to detect
        result = detect(image_url, topic, model, class_names)
        # Call the function to publish
        publish(result)
        # Wait for 60 seconds
        time.sleep(detect_interval)