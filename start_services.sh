#!/bin/bash
set -e

echo "Starting Unified Cloud Detection Service..."

# Run convert.py first to ensure model is converted
echo "Converting model if needed..."
python3 convert.py

# FIX: Exec into Python directly for clean signal handling
# Using exec replaces the shell with Python, enabling direct SIGTERM/SIGINT propagation
echo "Starting Alpaca SafetyMonitor (Waitress + MQTT unified service)..."
exec python3 alpaca_safety_monitor.py
