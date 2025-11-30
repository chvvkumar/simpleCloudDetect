# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -U pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container, excluding keras_model.h5 and labels.txt
COPY convert.py detect.py alpaca_safety_monitor.py start_services.sh ./

# Make the startup script executable
RUN chmod +x start_services.sh

# Run the startup script to launch both services
CMD ["./start_services.sh"]
