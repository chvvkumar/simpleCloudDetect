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
