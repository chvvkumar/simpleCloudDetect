# Use the official Python image from the Docker Hub
FROM python:3.11-slim as builder

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy application code
COPY convert.py detect.py alpaca_safety_monitor.py start_services.sh ./

# Make the startup script executable
RUN chmod +x start_services.sh

# Run the startup script to launch both services
CMD ["./start_services.sh"]
