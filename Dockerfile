# Use a lightweight Python base image
FROM python:3.11-slim-bookworm

# Set working directory
WORKDIR /app

# Install system dependencies (required for some Python packages)
# - build-essential: for compiling some python extensions
# - curl: for healthchecks
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
# Note: On ARM64 (Raspberry Pi), pip will automatically fetch the correct wheels
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY alpaca_safety_monitor.py .
COPY detect.py .
COPY train_model.py .

# Copy model and labels if they exist (will be mounted as volumes in production)
COPY --chown=root:root labels.txt* ./
COPY --chown=root:root model.onnx* ./

# Expose the ASCOM Alpaca port
EXPOSE 11111

# Healthcheck to ensure the web server is responding
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:11111/management/apiversions || exit 1

# Run the Alpaca Safety Monitor
CMD ["python", "alpaca_safety_monitor.py"]
