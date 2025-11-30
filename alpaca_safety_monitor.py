#!/usr/bin/env python3
"""
ASCOM Alpaca SafetyMonitor Server
Provides a safety monitor interface based on cloud detection
"""

import logging
import socket
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any
import json

from flask import Flask, request, jsonify
from flask_cors import CORS

from detect import CloudDetector, Config as DetectConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ASCOM Error Codes
ERROR_SUCCESS = 0
ERROR_NOT_IMPLEMENTED = 0x400  # 1024
ERROR_INVALID_VALUE = 0x401    # 1025
ERROR_NOT_CONNECTED = 0x407    # 1031

# Unsafe weather conditions
UNSAFE_CONDITIONS = {'Rain', 'Snow', 'Mostly Cloudy', 'Overcast'}


@dataclass
class AlpacaConfig:
    """Configuration for the Alpaca server"""
    port: int = 11111
    device_number: int = 0
    device_name: str = "Cloud Detection Safety Monitor"
    device_description: str = "ASCOM SafetyMonitor based on ML cloud detection"
    driver_info: str = "ASCOM Alpaca SafetyMonitor v1.0 - Cloud Detection Driver"
    driver_version: str = "1.0"
    interface_version: int = 1
    update_interval: int = 30  # seconds between cloud detection updates


class AlpacaSafetyMonitor:
    """ASCOM Alpaca SafetyMonitor implementation"""
    
    def __init__(self, alpaca_config: AlpacaConfig, detect_config: DetectConfig):
        self.alpaca_config = alpaca_config
        self.detect_config = detect_config
        self.server_transaction_id = 0
        self.transaction_lock = threading.Lock()
        
        # Device state
        self.connected = False
        self.connecting = False
        
        # Cloud detection state
        self.latest_detection: Optional[Dict[str, Any]] = None
        self.detection_lock = threading.Lock()
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_detection = threading.Event()
        
        # Initialize cloud detector (will be created when connected)
        self.cloud_detector: Optional[CloudDetector] = None
        
        logger.info(f"Initialized {self.alpaca_config.device_name}")
    
    def get_next_transaction_id(self) -> int:
        """Generate next server transaction ID (thread-safe)"""
        with self.transaction_lock:
            self.server_transaction_id += 1
            if self.server_transaction_id > 4294967295:
                self.server_transaction_id = 1
            return self.server_transaction_id
    
    def create_response(self, value: Any = None, error_number: int = ERROR_SUCCESS, 
                       error_message: str = "", client_transaction_id: int = 0) -> Dict[str, Any]:
        """Create standard ASCOM Alpaca response"""
        response = {
            "ClientTransactionID": client_transaction_id,
            "ServerTransactionID": self.get_next_transaction_id(),
            "ErrorNumber": error_number,
            "ErrorMessage": error_message
        }
        
        if value is not None or error_number == ERROR_SUCCESS:
            response["Value"] = value
        
        return response
    
    def get_client_params(self) -> tuple:
        """Extract client ID and transaction ID from request"""
        client_id = int(request.args.get('ClientID', 0) or request.form.get('ClientID', 0))
        client_transaction_id = int(request.args.get('ClientTransactionID', 0) or 
                                    request.form.get('ClientTransactionID', 0))
        return client_id, client_transaction_id
    
    def _detection_loop(self):
        """Background thread for continuous cloud detection"""
        logger.info("Starting cloud detection loop")
        
        while not self.stop_detection.is_set():
            try:
                result = self.cloud_detector.detect()
                with self.detection_lock:
                    self.latest_detection = result
                logger.info(f"Cloud detection: {result['class_name']} "
                           f"({result['confidence_score']:.1f}%)")
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
            
            # Wait for next update
            self.stop_detection.wait(self.alpaca_config.update_interval)
        
        logger.info("Detection loop stopped")
    
    def connect(self):
        """Connect to the device"""
        if self.connected:
            return
        
        try:
            # Initialize cloud detector
            self.cloud_detector = CloudDetector(self.detect_config)
            
            # Start detection loop
            self.stop_detection.clear()
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
            # Perform initial detection
            time.sleep(1)  # Give thread time to start
            
            self.connected = True
            logger.info("Connected to safety monitor")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from the device"""
        if not self.connected:
            return
        
        try:
            # Stop detection loop
            self.stop_detection.set()
            if self.detection_thread:
                self.detection_thread.join(timeout=5)
            
            self.connected = False
            self.cloud_detector = None
            logger.info("Disconnected from safety monitor")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def is_safe(self) -> bool:
        """Determine if conditions are safe based on latest detection"""
        with self.detection_lock:
            if self.latest_detection is None:
                # No detection yet - assume unsafe
                return False
            
            cloud_condition = self.latest_detection.get('class_name', '')
            is_safe = cloud_condition not in UNSAFE_CONDITIONS
            
            logger.debug(f"Safety check: {cloud_condition} -> {'SAFE' if is_safe else 'UNSAFE'}")
            return is_safe
    
    def get_device_state(self) -> list:
        """Get current operational state"""
        with self.detection_lock:
            state = [
                {"Name": "IsSafe", "Value": self.is_safe() if self.connected else False}
            ]
            
            if self.latest_detection:
                state.append({"Name": "CloudCondition", "Value": self.latest_detection.get('class_name', 'Unknown')})
                state.append({"Name": "Confidence", "Value": self.latest_detection.get('confidence_score', 0.0)})
                state.append({"Name": "LastUpdate", "Value": datetime.utcnow().isoformat() + 'Z'})
            
            return state


# Flask application setup
app = Flask(__name__)
CORS(app)

# Global safety monitor instance (will be initialized in main)
safety_monitor: Optional[AlpacaSafetyMonitor] = None


@app.route('/api/v1/safetymonitor/<int:device_number>/issafe', methods=['GET'])
def get_issafe(device_number: int):
    """Get safety status"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    if not safety_monitor.connected:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_NOT_CONNECTED,
            error_message="Device not connected",
            client_transaction_id=client_tx_id
        )), 200
    
    try:
        is_safe = safety_monitor.is_safe()
        return jsonify(safety_monitor.create_response(
            value=is_safe,
            client_transaction_id=client_tx_id
        ))
    except Exception as e:
        logger.error(f"Error getting safety status: {e}")
        return jsonify(safety_monitor.create_response(
            error_number=0x500,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500


@app.route('/api/v1/safetymonitor/<int:device_number>/connected', methods=['GET'])
def get_connected(device_number: int):
    """Get connection state"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.connected,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/connected', methods=['PUT'])
def set_connected(device_number: int):
    """Set connection state"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    try:
        connected = request.form.get('Connected', '').lower() == 'true'
        
        if connected and not safety_monitor.connected:
            safety_monitor.connect()
        elif not connected and safety_monitor.connected:
            safety_monitor.disconnect()
        
        return jsonify(safety_monitor.create_response(
            client_transaction_id=client_tx_id
        ))
    except Exception as e:
        logger.error(f"Error setting connection state: {e}")
        return jsonify(safety_monitor.create_response(
            error_number=0x500,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500


@app.route('/api/v1/safetymonitor/<int:device_number>/connecting', methods=['GET'])
def get_connecting(device_number: int):
    """Get connecting state (Platform 7)"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.connecting,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/description', methods=['GET'])
def get_description(device_number: int):
    """Get device description"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.alpaca_config.device_description,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/devicestate', methods=['GET'])
def get_devicestate(device_number: int):
    """Get device state (Platform 7)"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    try:
        state = safety_monitor.get_device_state()
        return jsonify(safety_monitor.create_response(
            value=state,
            client_transaction_id=client_tx_id
        ))
    except Exception as e:
        logger.error(f"Error getting device state: {e}")
        return jsonify(safety_monitor.create_response(
            error_number=0x500,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500


@app.route('/api/v1/safetymonitor/<int:device_number>/driverinfo', methods=['GET'])
def get_driverinfo(device_number: int):
    """Get driver info"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.alpaca_config.driver_info,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/driverversion', methods=['GET'])
def get_driverversion(device_number: int):
    """Get driver version"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.alpaca_config.driver_version,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/interfaceversion', methods=['GET'])
def get_interfaceversion(device_number: int):
    """Get interface version"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.alpaca_config.interface_version,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/name', methods=['GET'])
def get_name(device_number: int):
    """Get device name"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.alpaca_config.device_name,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/supportedactions', methods=['GET'])
def get_supportedactions(device_number: int):
    """Get supported actions"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    # No custom actions supported
    return jsonify(safety_monitor.create_response(
        value=[],
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/action', methods=['PUT'])
def put_action(device_number: int):
    """Execute custom action (not implemented)"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    return jsonify(safety_monitor.create_response(
        error_number=ERROR_NOT_IMPLEMENTED,
        error_message="No custom actions are supported",
        client_transaction_id=client_tx_id
    )), 200


@app.route('/api/v1/safetymonitor/<int:device_number>/commandblind', methods=['PUT'])
@app.route('/api/v1/safetymonitor/<int:device_number>/commandbool', methods=['PUT'])
@app.route('/api/v1/safetymonitor/<int:device_number>/commandstring', methods=['PUT'])
def deprecated_commands(device_number: int):
    """Deprecated command methods"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    return jsonify(safety_monitor.create_response(
        error_number=ERROR_NOT_IMPLEMENTED,
        error_message="This method is deprecated",
        client_transaction_id=client_tx_id
    )), 200


@app.route('/management/apiversions', methods=['GET'])
def get_apiversions():
    """Get supported API versions"""
    return jsonify([1])


@app.route('/management/v1/description', methods=['GET'])
def get_management_description():
    """Get server description"""
    return jsonify({
        "ServerName": "Cloud Detection Safety Monitor",
        "Manufacturer": "Open Source",
        "ManufacturerVersion": "1.0",
        "Location": "Cloud"
    })


@app.route('/management/v1/configureddevices', methods=['GET'])
def get_configured_devices():
    """Get list of configured devices"""
    return jsonify([{
        "DeviceName": safety_monitor.alpaca_config.device_name,
        "DeviceType": "SafetyMonitor",
        "DeviceNumber": safety_monitor.alpaca_config.device_number,
        "UniqueID": "cloud-safety-monitor-0"
    }])


class AlpacaDiscovery:
    """ASCOM Alpaca UDP Discovery Protocol Handler"""
    
    DISCOVERY_PORT = 32227
    DISCOVERY_MESSAGE = b"alpacadiscovery1"
    
    def __init__(self, alpaca_port: int):
        self.alpaca_port = alpaca_port
        self.socket = None
        self.running = False
        self.thread = None
    
    def start(self):
        """Start the discovery service"""
        try:
            # Create UDP socket
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            
            # Bind to all interfaces on discovery port
            self.socket.bind(('', self.DISCOVERY_PORT))
            
            self.running = True
            self.thread = threading.Thread(target=self._discovery_loop, daemon=True)
            self.thread.start()
            
            logger.info(f"Alpaca Discovery service started on UDP port {self.DISCOVERY_PORT}")
        except Exception as e:
            logger.error(f"Failed to start discovery service: {e}")
            logger.warning("Discovery will not be available, but HTTP API will still work")
    
    def stop(self):
        """Stop the discovery service"""
        self.running = False
        if self.socket:
            self.socket.close()
        if self.thread:
            self.thread.join(timeout=2)
        logger.info("Alpaca Discovery service stopped")
    
    def _discovery_loop(self):
        """Main discovery loop"""
        logger.info("Discovery loop started, waiting for discovery requests...")
        
        while self.running:
            try:
                # Receive discovery request
                data, addr = self.socket.recvfrom(1024)
                
                if data == self.DISCOVERY_MESSAGE:
                    logger.info(f"Discovery request from {addr[0]}:{addr[1]}")
                    self._send_discovery_response(addr)
                else:
                    logger.debug(f"Ignored non-discovery message from {addr}: {data}")
                    
            except Exception as e:
                if self.running:  # Only log if we're supposed to be running
                    logger.error(f"Error in discovery loop: {e}")
    
    def _send_discovery_response(self, addr):
        """Send discovery response to client"""
        try:
            # Build discovery response per ASCOM spec
            response = {
                "AlpacaPort": self.alpaca_port
            }
            
            response_json = json.dumps(response).encode('utf-8')
            self.socket.sendto(response_json, addr)
            logger.info(f"Sent discovery response to {addr[0]}:{addr[1]} - Port: {self.alpaca_port}")
            
        except Exception as e:
            logger.error(f"Failed to send discovery response: {e}")


# Initialize application at module level for gunicorn
def create_app():
    """Application factory for gunicorn"""
    import os
    
    global safety_monitor
    
    # Load detection configuration
    detect_config = DetectConfig.from_env()
    
    # Create Alpaca configuration
    alpaca_config = AlpacaConfig(
        port=int(os.getenv('ALPACA_PORT', '11111')),
        device_number=int(os.getenv('ALPACA_DEVICE_NUMBER', '0')),
        update_interval=int(os.getenv('ALPACA_UPDATE_INTERVAL', '30'))
    )
    
    # Initialize safety monitor
    safety_monitor = AlpacaSafetyMonitor(alpaca_config, detect_config)
    
    # Initialize and start discovery service
    discovery = AlpacaDiscovery(alpaca_config.port)
    discovery.start()
    
    logger.info(f"ASCOM Alpaca SafetyMonitor initialized")
    logger.info(f"Device: {alpaca_config.device_name}")
    logger.info(f"Update interval: {alpaca_config.update_interval}s")
    logger.info(f"Unsafe conditions: {', '.join(UNSAFE_CONDITIONS)}")
    
    return app


def main():
    """Main entry point for standalone execution"""
    import os
    
    # Create and configure app
    application = create_app()
    
    # Get port from environment
    port = int(os.getenv('ALPACA_PORT', '11111'))
    
    logger.info(f"Starting ASCOM Alpaca SafetyMonitor on port {port}")
    
    # Run Flask development server
    application.run(host='0.0.0.0', port=port, debug=False)


if __name__ == '__main__':
    main()
else:
    # When imported by gunicorn, initialize the app
    app = create_app()
