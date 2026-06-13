#!/usr/bin/env python3
"""
Main entry point for ASCOM Alpaca SafetyMonitor Server
"""
import threading
import signal
import sys
import logging
from waitress import serve
from alpaca import create_app
from alpaca.discovery import AlpacaDiscovery

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main entry point for Waitress-based unified service"""
    app, safety_monitor = create_app()
    port = safety_monitor.alpaca_config.port
    
    # Start Background Detection Thread
    safety_monitor.stop_detection.clear()
    safety_monitor.detection_thread = threading.Thread(
        target=safety_monitor._detection_loop, daemon=True
    )
    safety_monitor.detection_thread.start()
    logger.info("Detection loop started")
    
    # Start UDP Discovery Service
    discovery = AlpacaDiscovery(port)
    discovery.start()
    
    # Setup graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutdown initiated...")
        discovery.stop()
        safety_monitor.stop_detection.set()
        safety_monitor.disconnect()
        sys.exit(0)
        
    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    
    # Start HTTP Server
    logger.info(f"ASCOM Alpaca SafetyMonitor starting on port {port}")
    logger.info(f"Web UI: http://localhost:{port}/setup/v1/safetymonitor/0/setup")
    serve(app, host='0.0.0.0', port=port, threads=32)

if __name__ == "__main__":
    main()
