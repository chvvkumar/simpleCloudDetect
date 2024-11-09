# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Install curl
RUN apt-get update && apt-get install -y curl

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the MQTT broker will use
EXPOSE 1883

# Define environment variables
ENV image_url="https://allsky.challa.co:1982/current/resized/image.jpg"
ENV BROKER="192.168.1.250"
ENV PORT=1883
ENV TOPIC="Astro/DockerSimpleCloudDetect"
ENV DETECT_INTERVAL=15

# Run the application
CMD ["python", "detect.py"]