#!/bin/bash
set -e

echo "Starting Cloud Detection Services..."

# Run convert.py first to ensure model is converted
echo "Converting model if needed..."
python convert.py

echo "Starting MQTT cloud detection service in background..."
python detect.py &
DETECT_PID=$!

echo "Starting Alpaca SafetyMonitor API server..."
gunicorn alpaca_safety_monitor:app \
    --bind 0.0.0.0:${ALPACA_PORT:-11111} \
    --workers 1 \
    --threads 8 \
    --timeout 120 \
    --keep-alive 30 \
    --graceful-timeout 10 \
    --worker-class gthread \
    --log-level info &
ALPACA_PID=$!

echo "Services started:"
echo "  - Cloud Detection (MQTT): PID $DETECT_PID"
echo "  - Alpaca SafetyMonitor: PID $ALPACA_PID (port ${ALPACA_PORT:-11111})"

# Wait for both processes
wait -n

# If either process exits, exit the script
exit $?
