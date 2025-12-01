#!/usr/bin/env python3
"""
ASCOM Alpaca SafetyMonitor Server
Provides a safety monitor interface based on SimpleCloudDetect cloud detection:
https://github.com/chvvkumar/simpleCloudDetect

Authors: chvvkumar
"""

import logging
import socket
import threading
import time
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any
import json

from flask import Flask, request, jsonify, render_template_string
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
ERROR_UNSPECIFIED = 0x500      # 1280

# Available cloud conditions from ML model
ALL_CLOUD_CONDITIONS = ['Clear', 'Mostly Cloudy', 'Overcast', 'Rain', 'Snow', 'Wisps of clouds']


@dataclass
class AlpacaConfig:
    """Configuration for the Alpaca server"""
    port: int = 11111
    device_number: int = 0
    device_name: str = "SimpleCloudDetect"
    device_description: str = "ASCOM SafetyMonitor based on ML cloud detection"
    driver_info: str = "ASCOM Alpaca SafetyMonitor v2.0 - Cloud Detection Driver"
    driver_version: str = "2.0"
    interface_version: int = 3  # Changed from 1 to 3
    update_interval: int = 30  # seconds between cloud detection updates
    location: str = "AllSky Camera"
    unsafe_conditions: list = field(default_factory=lambda: ['Rain', 'Snow', 'Mostly Cloudy', 'Overcast'])
    
    def save_to_file(self, filepath: str = "alpaca_config.json"):
        """Save configuration to JSON file"""
        try:
            config_dict = asdict(self)
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
    
    @classmethod
    def load_from_file(cls, filepath: str = "alpaca_config.json"):
        """Load configuration from JSON file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)
                logger.info(f"Configuration loaded from {filepath}")
                return cls(**config_dict)
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
        return None


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
        
        # Pre-load cloud detector at startup to avoid delays during connection
        logger.info("Pre-loading ML model...")
        self.cloud_detector = CloudDetector(self.detect_config)
        logger.info("ML model loaded successfully")
        
        # Perform initial detection to populate latest_detection BEFORE connection
        try:
            logger.info("Performing initial detection...")
            initial_result = self.cloud_detector.detect()
            with self.detection_lock:
                self.latest_detection = initial_result
            logger.info(f"Initial detection complete: {initial_result['class_name']}")
        except Exception as e:
            logger.error(f"Initial detection failed: {e}")
            # Set a safe default state
            with self.detection_lock:
                self.latest_detection = {
                    'class_name': 'Unknown',
                    'confidence_score': 0.0,
                    'Detection Time (Seconds)': 0.0
                }
        
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
        """Extract client ID and transaction ID from request with validation"""
        # Use case-insensitive retrieval
        client_id_raw = self._get_arg('ClientID', '0')
        client_tx_raw = self._get_arg('ClientTransactionID', '0')
        
        # Strip whitespace and convert to string
        client_id_raw = str(client_id_raw).strip()
        client_tx_raw = str(client_tx_raw).strip()
        
        # Parse ClientID with fallback to 0
        try:
            client_id = int(client_id_raw) if client_id_raw else 0
        except (ValueError, TypeError):
            logger.warning(f"Invalid ClientID received: {client_id_raw}")
            client_id = 0
        
        # Parse ClientTransactionID with fallback to 0
        try:
            client_transaction_id = int(client_tx_raw) if client_tx_raw else 0
            
            # Validate ClientTransactionID is unsigned 32-bit (0 to 4294967295)
            # Per ASCOM Alpaca spec, this must be a uint32
            if client_transaction_id < 0 or client_transaction_id > 4294967295:
                logger.warning(f"ClientTransactionID out of range: {client_transaction_id}")
                client_transaction_id = 0
        except (ValueError, TypeError):
            logger.warning(f"Invalid ClientTransactionID received: {client_tx_raw}")
            client_transaction_id = 0
            
        return client_id, client_transaction_id
    
    def _detection_loop(self):
        """Background thread for continuous cloud detection"""
        logger.info("Starting cloud detection loop")
        
        while not self.stop_detection.is_set():
            try:
                # Perform detection in background without blocking API responses
                result = self.cloud_detector.detect()
                with self.detection_lock:
                    self.latest_detection = result
                logger.info(f"Cloud detection: {result['class_name']} "
                           f"({result['confidence_score']:.1f}%)")
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                # On error, keep previous detection state
            
            # Wait for next update - use shorter intervals to prevent worker timeout
            # Break waiting into 1-second chunks so we can respond to stop signal
            for _ in range(self.alpaca_config.update_interval):
                if self.stop_detection.wait(1.0):  # Wait 1 second at a time
                    break
        
        logger.info("Detection loop stopped")
    
    def connect(self):
        """Connect to the device"""
        if self.connected:
            return
        
        try:
            # Start detection loop (model is already pre-loaded)
            self.stop_detection.clear()
            self.detection_thread = threading.Thread(target=self._detection_loop, daemon=True)
            self.detection_thread.start()
            
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
            # Signal detection loop to stop (non-blocking)
            self.stop_detection.set()
            # Don't wait for thread to join - let it stop asynchronously
            # Thread is daemon so it will terminate when program exits
            
            self.connected = False
            logger.info("Disconnected from safety monitor")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def is_safe(self) -> bool:
        """Determine if conditions are safe based on latest detection"""
        with self.detection_lock:
            if self.latest_detection is None:
                # No detection yet - assume unsafe for safety
                logger.warning("No detection data available, returning unsafe")
                return False
            
            cloud_condition = self.latest_detection.get('class_name', '')
            
            # If detection confidence is very low or unknown, treat as unsafe
            confidence = self.latest_detection.get('confidence_score', 0.0)
            if confidence < 50.0 or cloud_condition == 'Unknown':
                logger.warning(f"Low confidence or unknown condition: {cloud_condition} ({confidence}%), returning unsafe")
                return False
            
            is_safe = cloud_condition not in self.alpaca_config.unsafe_conditions
            
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
        
    def _get_arg(self, key: str, default: Any = None) -> str:
        """Case-insensitive argument retrieval from request values"""
        # Search in both query args and form data (combined in request.values)
        key_lower = key.lower()
        for k, v in request.values.items():
            if k.lower() == key_lower:
                return v
        return default


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
    
    # Per ASCOM spec: Always return a value, never an error
    # If disconnected, return False (unsafe) to protect equipment
    if not safety_monitor.connected:
        return jsonify(safety_monitor.create_response(
            value=False,  # Always unsafe when disconnected
            client_transaction_id=client_tx_id
        ))
    
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

    # Strict Boolean Validation
    connected_str = safety_monitor._get_arg('Connected', '').strip().lower()
    
    if connected_str == 'true':
        target_state = True
    elif connected_str == 'false':
        target_state = False
    else:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid boolean value for Connected: '{connected_str}'",
            client_transaction_id=client_tx_id
        )), 400

    try:
        if target_state and not safety_monitor.connected:
            safety_monitor.connect()
        elif not target_state and safety_monitor.connected:
            safety_monitor.disconnect()
        
        return jsonify(safety_monitor.create_response(
            client_transaction_id=client_tx_id
        ))
    except Exception as e:
        logger.error(f"Failed to set connected state: {e}")
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_UNSPECIFIED,
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


@app.route('/api/v1/safetymonitor/<int:device_number>/connect', methods=['PUT'])
def connect_device(device_number: int):
    """Connect to device asynchronously (Platform 7, Interface V3+)"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    try:
        if not safety_monitor.connected:
            safety_monitor.connect()
        
        return jsonify(safety_monitor.create_response(
            client_transaction_id=client_tx_id
        ))
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_UNSPECIFIED,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500


@app.route('/api/v1/safetymonitor/<int:device_number>/disconnect', methods=['PUT'])
def disconnect_device(device_number: int):
    """Disconnect from device asynchronously (Platform 7, Interface V3+)"""
    _, client_tx_id = safety_monitor.get_client_params()
    
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    try:
        if safety_monitor.connected:
            safety_monitor.disconnect()
        
        return jsonify(safety_monitor.create_response(
            client_transaction_id=client_tx_id
        ))
    except Exception as e:
        logger.error(f"Error during disconnect: {e}")
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_UNSPECIFIED,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500


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
    return jsonify({
        "Value": [1],
        "ClientTransactionID": 0,
        "ServerTransactionID": safety_monitor.get_next_transaction_id(),
        "ErrorNumber": 0,
        "ErrorMessage": ""
    })


@app.route('/management/v1/description', methods=['GET'])
def get_management_description():
    """Get server description"""
    value = {
        "ServerName": safety_monitor.alpaca_config.device_name,
        "Manufacturer": "chvvkumar",
        "ManufacturerVersion": "1.0",
        "Location": safety_monitor.alpaca_config.location
    }
    # MUST wrap in create_response
    return jsonify(safety_monitor.create_response(value=value))


@app.route('/management/v1/configureddevices', methods=['GET'])
def get_configured_devices():
    """Get list of configured devices"""
    value = [{
        "DeviceName": safety_monitor.alpaca_config.device_name,
        "DeviceType": "SafetyMonitor",
        "DeviceNumber": safety_monitor.alpaca_config.device_number,
        "UniqueID": "cloud-safety-monitor-0"
    }]
    # MUST wrap in create_response
    return jsonify(safety_monitor.create_response(value=value))


def get_available_cloud_conditions() -> list:
    """Load cloud conditions from labels file"""
    try:
        label_path = os.getenv('LABEL_PATH', 'labels.txt')
        with open(label_path, 'r') as f:
            # Parse labels like "0 Clear" and extract just the class name
            return [line.strip().split(' ', 1)[1] for line in f.readlines()]
    except Exception as e:
        logger.error(f"Failed to load labels: {e}")
        # Fallback to hardcoded list
        return ['Clear', 'Mostly Cloudy', 'Overcast', 'Rain', 'Snow', 'Wisps of clouds']


@app.route('/setup/v1/safetymonitor/<int:device_number>/setup', methods=['GET', 'POST'])
def setup_device(device_number: int):
    """Setup page for configuring device name and location"""
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify({"error": "Invalid device number"}), 404
    
    if request.method == 'POST':
        # Handle form submission
        device_name = request.form.get('device_name', '').strip()
        location = request.form.get('location', '').strip()
        
        if device_name:
            safety_monitor.alpaca_config.device_name = device_name
            logger.info(f"Device name updated to: {device_name}")
        
        if location:
            safety_monitor.alpaca_config.location = location
            logger.info(f"Location updated to: {location}")
        
        # Handle unsafe conditions checkboxes
        unsafe_conditions = []
        all_conditions = get_available_cloud_conditions()  # Use dynamic conditions instead of ALL_CLOUD_CONDITIONS
        for condition in all_conditions:
            if request.form.get(f'unsafe_{condition}'):
                unsafe_conditions.append(condition)
        
        safety_monitor.alpaca_config.unsafe_conditions = unsafe_conditions
        logger.info(f"Unsafe conditions updated to: {unsafe_conditions}")
        
        # Save configuration to file
        safety_monitor.alpaca_config.save_to_file()
        
        # Return success message
        message = "Configuration updated successfully!"
    else:
        message = ""
    
    # HTML form for setup
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SimpleCloudDetect Setup</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 700px;
                margin: 50px auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            h1 {
                color: #333;
                border-bottom: 2px solid #4CAF50;
                padding-bottom: 10px;
            }
            h2 {
                color: #555;
                margin-top: 30px;
                margin-bottom: 15px;
                font-size: 18px;
            }
            .form-group {
                margin-bottom: 20px;
            }
            label {
                display: block;
                margin-bottom: 5px;
                color: #555;
                font-weight: bold;
            }
            input[type="text"] {
                width: 100%;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-sizing: border-box;
                font-size: 14px;
            }
            input[type="text"]:focus {
                outline: none;
                border-color: #4CAF50;
            }
            .checkbox-group {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 10px;
                padding: 15px;
                background-color: #f9f9f9;
                border-radius: 4px;
                border: 1px solid #ddd;
            }
            .checkbox-item {
                display: flex;
                align-items: center;
            }
            .checkbox-item input[type="checkbox"] {
                margin-right: 8px;
                width: 18px;
                height: 18px;
                cursor: pointer;
            }
            .checkbox-item label {
                margin: 0;
                font-weight: normal;
                cursor: pointer;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 12px 30px;
                border: none;
                border-radius: 4px;
                cursor: pointer;
                font-size: 16px;
                width: 100%;
                margin-top: 20px;
            }
            button:hover {
                background-color: #45a049;
            }
            .message {
                background-color: #d4edda;
                color: #155724;
                padding: 12px;
                border-radius: 4px;
                margin-bottom: 20px;
                border: 1px solid #c3e6cb;
            }
            .info {
                background-color: #e7f3ff;
                padding: 15px;
                border-radius: 4px;
                margin-bottom: 20px;
                border-left: 4px solid #2196F3;
            }
            .info p {
                margin: 5px 0;
                color: #555;
            }
            .help-text {
                font-size: 13px;
                color: #666;
                margin-top: 5px;
                font-style: italic;
            }
            .safe-indicator {
                color: #4CAF50;
                font-weight: bold;
            }
            .unsafe-indicator {
                color: #f44336;
                font-weight: bold;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>SimpleCloudDetect Setup</h1>
            
            {% if message %}
            <div class="message">{{ message }}</div>
            {% endif %}
            
            <div class="info">
                <p><strong>Current Configuration:</strong></p>
                <p>Device Name: {{ current_name }}</p>
                <p>Location: {{ current_location }}</p>
                <p>Safe Conditions: <span class="safe-indicator">{{ safe_conditions|join(', ') }}</span></p>
                <p>Unsafe Conditions: <span class="unsafe-indicator">{{ unsafe_conditions|join(', ') }}</span></p>
            </div>
            
            <form method="POST">
                <div class="form-group">
                    <label for="device_name">Device Name:</label>
                    <input type="text" id="device_name" name="device_name" 
                           value="{{ current_name }}" placeholder="Enter device name">
                </div>
                
                <div class="form-group">
                    <label for="location">Location:</label>
                    <input type="text" id="location" name="location" 
                           value="{{ current_location }}" placeholder="Enter location">
                </div>
                
                <div class="form-group">
                    <h2>Safety Configuration</h2>
                    <label>Mark conditions that are UNSAFE for observing:</label>
                    <div class="help-text">Unchecked conditions will be considered SAFE</div>
                    <div class="checkbox-group">
                        {% for condition in all_conditions %}
                        <div class="checkbox-item">
                            <input type="checkbox" 
                                   id="unsafe_{{ condition }}" 
                                   name="unsafe_{{ condition }}"
                                   {% if condition in unsafe_conditions %}checked{% endif %}>
                            <label for="unsafe_{{ condition }}">{{ condition }}</label>
                        </div>
                        {% endfor %}
                    </div>
                </div>
                
                <button type="submit">Save Configuration</button>
            </form>
        </div>
    </body>
    </html>
    """
    
    # Calculate safe vs unsafe conditions for display
    unsafe_conditions = safety_monitor.alpaca_config.unsafe_conditions
    safe_conditions = [c for c in ALL_CLOUD_CONDITIONS if c not in unsafe_conditions]
    
    return render_template_string(
        html_template,
        message=message,
        current_name=safety_monitor.alpaca_config.device_name,
        current_location=safety_monitor.alpaca_config.location,
        all_conditions=ALL_CLOUD_CONDITIONS,
        unsafe_conditions=unsafe_conditions,
        safe_conditions=safe_conditions
    )


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
    
    # Try to load Alpaca configuration from file first
    alpaca_config = AlpacaConfig.load_from_file()
    
    if alpaca_config is None:
        # Create default Alpaca configuration if file doesn't exist
        alpaca_config = AlpacaConfig(
            port=int(os.getenv('ALPACA_PORT', '11111')),
            device_number=int(os.getenv('ALPACA_DEVICE_NUMBER', '0')),
            update_interval=int(os.getenv('ALPACA_UPDATE_INTERVAL', '30'))
        )
        # Save default config
        alpaca_config.save_to_file()
    else:
        # Override port settings from environment if provided
        alpaca_config.port = int(os.getenv('ALPACA_PORT', str(alpaca_config.port)))
        alpaca_config.device_number = int(os.getenv('ALPACA_DEVICE_NUMBER', str(alpaca_config.device_number)))
        alpaca_config.update_interval = int(os.getenv('ALPACA_UPDATE_INTERVAL', str(alpaca_config.update_interval)))
    
    # Initialize safety monitor
    safety_monitor = AlpacaSafetyMonitor(alpaca_config, detect_config)
    
    # Initialize and start discovery service
    discovery = AlpacaDiscovery(alpaca_config.port)
    discovery.start()
    
    logger.info(f"ASCOM Alpaca SafetyMonitor initialized")
    logger.info(f"Device: {alpaca_config.device_name}")
    logger.info(f"Update interval: {alpaca_config.update_interval}s")
    logger.info(f"Unsafe conditions: {', '.join(alpaca_config.unsafe_conditions)}")
    logger.info(f"Safe conditions: {', '.join([c for c in ALL_CLOUD_CONDITIONS if c not in alpaca_config.unsafe_conditions])}")
    
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
