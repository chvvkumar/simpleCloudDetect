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
    if [ "$TARGETPLATFORM" = "linux/arm64" ]; \
    then \
        pip install --no-cache-dir -r requirements-arm64.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

# Final stage
FROM python:3.11-slim

WORKDIR /app

# Install curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    dos2unix \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd -m -u 1000 appuser

# Copy Python packages and binaries from builder
# NOTE: We generally do NOT need to chown site-packages; read-only access is sufficient for the app.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code with ownership set explicitly during copy
# This avoids the expensive "chown -R" layer later
COPY --chown=appuser:appuser convert.py detect.py main.py start_services.sh ./
COPY --chown=appuser:appuser alpaca/ ./alpaca/
COPY --chown=appuser:appuser templates/ ./templates/

# Fix line endings and make the startup script executable
# We only chown the specific script we modified if necessary, not the whole /app recursively
RUN dos2unix start_services.sh && \
    chmod +x start_services.sh && \
    chown appuser:appuser start_services.sh

# Switch to non-root user
USER appuser

# FIX: Add healthcheck to ensure container restarts if Python process hangs
HEALTHCHECK --interval=60s --timeout=10s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:${ALPACA_PORT:-11111}/api/v1/safetymonitor/${ALPACA_DEVICE_NUMBER:-0}/connected || exit 1

# Run the startup script to launch unified service
CMD ["./start_services.sh"]