# Use the official Python image from the Docker Hub
FROM python:3.14-slim as builder

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements files
COPY requirements.txt requirements-arm64.txt ./

# Install dependencies based on architecture
ARG TARGETPLATFORM
RUN pip install --no-cache-dir --upgrade pip && \
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; then \
        pip install --no-cache-dir -r requirements-arm64.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Final stage
FROM python:3.14-slim

WORKDIR /app

# Copy Python packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY convert.py detect.py alpaca_safety_monitor.py start_services.sh ./

# Make the startup script executable
RUN chmod +x start_services.sh

# Run the startup script to launch both services
CMD ["./start_services.sh"]
