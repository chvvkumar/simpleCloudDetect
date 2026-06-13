#!/bin/bash
set -e

echo "Starting Unified Cloud Detection Service..."

# FIX: Exec into Python directly for clean signal handling
# Using exec replaces the shell with Python, enabling direct SIGTERM/SIGINT propagation
echo "Starting Alpaca SafetyMonitor (Waitress + MQTT unified service)..."
exec python3 main.py
