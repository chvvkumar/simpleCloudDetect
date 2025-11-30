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
gunicorn --bind 0.0.0.0:11111 --workers 1 --threads 4 --timeout 120 alpaca_safety_monitor:app &
ALPACA_PID=$!

echo "Services started:"
echo "  - Cloud Detection (MQTT): PID $DETECT_PID"
echo "  - Alpaca SafetyMonitor: PID $ALPACA_PID (port 11111)"

# Wait for both processes
wait -n

# If either process exits, exit the script
exit $?
