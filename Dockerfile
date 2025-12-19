FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy App
COPY convert.py detect.py alpaca_safety_monitor.py start_services.sh ./

# Fix line endings and make the startup script executable and set ownership
RUN dos2unix start_services.sh && \
    chmod +x start_services.sh && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# FIX: Add healthcheck to ensure container restarts if Python process hangs
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${ALPACA_PORT:-11111}/api/v1/safetymonitor/${ALPACA_DEVICE_NUMBER:-0}/connected || exit 1

# Run the startup script to launch unified service
CMD ["./start_services.sh"]
