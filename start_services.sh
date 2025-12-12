#!/bin/bash
set -e

echo "Starting Unified Cloud Detection Service..."

# Run convert.py first to ensure model is converted
echo "Converting model if needed..."
python convert.py

# Run unified service (ASCOM API + MQTT in single process)
echo "Starting Alpaca SafetyMonitor API server (with embedded MQTT)..."
exec gunicorn alpaca_safety_monitor:app \
    --bind 0.0.0.0:${ALPACA_PORT:-11111} \
    --workers 1 \
    --threads 8 \
    --timeout 120 \
    --keep-alive 30 \
    --graceful-timeout 10 \
    --worker-class gthread \
    --log-level info
