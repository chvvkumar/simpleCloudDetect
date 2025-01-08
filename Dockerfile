# Use Python slim-bullseye as base
FROM python:3.11-slim-bullseye as builder

# Set working directory
WORKDIR /app

# Copy only requirements file first to leverage Docker cache
COPY requirements.txt .

# Install build dependencies and create virtual environment
RUN python -m venv /opt/venv && \
    /opt/venv/bin/pip install --no-cache-dir -U pip && \
    /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

# Final stage
FROM python:3.11-slim-bullseye

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv

# Set working directory
WORKDIR /app

# Copy application files
COPY convert.py detect.py keras_model.h5 labels.txt ./

# Set proper permissions
RUN chown clouddetect:clouddetect /app/keras_model.h5 && \
    chmod 644 /app/keras_model.h5

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Create non-root user
RUN useradd -m -r clouddetect && \
    chown -R clouddetect:clouddetect /app

# Switch to non-root user
USER clouddetect

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get(os.environ['IMAGE_URL'])" || exit 1

# Run convert.py first, then detect.py
CMD ["sh", "-c", "python convert.py && python detect.py"]
