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

# Run the application
CMD ["python", "detect.py"]