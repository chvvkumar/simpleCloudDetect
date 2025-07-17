# Multi-architecture Dockerfile for SimpleCloudDetect
# Supports x86_64 (amd64) and ARM (Raspberry Pi)

ARG PYTHON_VERSION=3.11
ARG DEBIAN_FRONTEND=noninteractive

# Base stage with common dependencies
FROM python:${PYTHON_VERSION}-slim AS base

# Set the working directory in the container
WORKDIR /app

# Copy shared requirements
COPY requirements-common.txt .

# Install common dependencies
RUN pip install -U pip && \
    pip install --no-cache-dir -r requirements-common.txt

# Add labels
LABEL maintainer="simpleCloudDetect" \
      description="Cloud detection service with multi-architecture support"

# AMD64 (x86_64) specific stage
FROM base AS amd64
COPY requirements-amd64.txt .
RUN pip install --no-cache-dir -r requirements-amd64.txt

# ARM specific stage
FROM base AS arm
COPY requirements-arm.txt .
# Use the pre-compiled tflite-runtime which is optimized for ARM
RUN pip install --no-cache-dir -r requirements-arm.txt

# Final stage that selects the appropriate base image
FROM ${TARGETARCH:-amd64}

# Copy all application files
COPY convert.py detect.py keras_model.h5 labels.txt ./

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Run convert.py first, then detect.py
CMD ["sh", "-c", "python convert.py && python detect.py"]
