# Use the official Python image from the Docker Hub
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install -U pip & pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Run convert.py first, then detect.py
CMD ["sh", "-c", "python convert.py && python detect.py"]
