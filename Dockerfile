# Use the official Python image from the Docker Hub
FROM python:3.11-slim AS builder

# Set the working directory
WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Copy ONLY requirements files first for better layer caching
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
FROM python:3.11-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy Python packages and binaries from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code (this layer changes frequently, so it's last)
COPY convert.py detect.py alpaca_safety_monitor.py start_services.sh ./

# Make the startup script executable and set ownership
RUN chmod +x start_services.sh && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Run the startup script to launch unified service
CMD ["./start_services.sh"]
