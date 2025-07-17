# Multi-architecture Dockerfile for SimpleCloudDetect
# Supports x86_64 (amd64) and ARM (Raspberry Pi)

FROM python:3.11-slim

ARG TARGETARCH
RUN echo "Building for architecture: ${TARGETARCH}"

# Set the working directory in the container
WORKDIR /app

# Copy requirements files
COPY requirements-common.txt requirements-amd64.txt requirements-arm.txt ./

# Install common dependencies first
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements-common.txt

# Install architecture-specific dependencies
RUN if [ "$TARGETARCH" = "arm64" ] || [ "$TARGETARCH" = "arm" ]; then \
        echo "Installing ARM dependencies" && \
        pip install --no-cache-dir -r requirements-arm.txt; \
    else \
        echo "Installing AMD64 dependencies" && \
        pip install --no-cache-dir -r requirements-amd64.txt; \
    fi

# Copy application files
COPY convert.py detect.py keras_model.h5 labels.txt ./

# Add labels
LABEL maintainer="simpleCloudDetect" \
      description="Cloud detection service with multi-architecture support" \
      architecture="${TARGETARCH}"

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run convert.py first, then detect.py
CMD ["sh", "-c", "python convert.py && python detect.py"]
