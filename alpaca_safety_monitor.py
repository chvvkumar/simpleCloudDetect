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
import signal
import sys
from dataclasses import dataclass, asdict, field
from datetime import datetime
from typing import Optional, Dict, Any
import json

from flask import Flask, request, jsonify, render_template_string, redirect, url_for
from flask_cors import CORS
import paho.mqtt.client as mqtt

from detect import CloudDetector, Config as DetectConfig, HADiscoveryManager

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
        self.connected_at: Optional[datetime] = None
        
        # Cloud detection state - use single lock for all state
        self.latest_detection: Optional[Dict[str, Any]] = {
            'class_name': 'Unknown',
            'confidence_score': 0.0,
            'Detection Time (Seconds)': 0.0,
            'timestamp': None
        }
        self._cached_is_safe = False  # Cache safe status to reduce lock contention
        self._unsafe_conditions_set = set(alpaca_config.unsafe_conditions)  # Set lookup is O(1)
        self.detection_lock = threading.Lock()
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_detection = threading.Event()
        
        # Setup MQTT client first
        self.mqtt_client = self._setup_mqtt()
        self.ha_discovery = None
        
        # Pre-load cloud detector at startup (pass MQTT client to avoid double connection)
        logger.info("Pre-loading ML model...")
        self.cloud_detector = CloudDetector(self.detect_config, mqtt_client=self.mqtt_client)
        logger.info("ML model loaded successfully")
        
        # Setup HA Discovery if enabled
        if self.mqtt_client and self.detect_config.mqtt_discovery_mode == 'homeassistant':
            self.ha_discovery = HADiscoveryManager(self.detect_config, self.mqtt_client)
            self.ha_discovery.publish_discovery_configs()
        
        # FIX: Blocking initial detection to ensure readiness (better than background thread)
        logger.info("Performing initial detection (blocking)...")
        try:
            initial_result = self.cloud_detector.detect()
            initial_result['timestamp'] = datetime.now()
            with self.detection_lock:
                self.latest_detection = initial_result
                self._update_cached_safety(initial_result)
            logger.info(f"Initial detection complete: {initial_result['class_name']}")
        except Exception as e:
            logger.error(f"Initial detection failed: {e}")
        
        logger.info(f"Initialized {self.alpaca_config.device_name}")
    
    def _update_cached_safety(self, detection: Dict[str, Any]):
        """Update cached safety status (assumes lock is held)"""
        cloud_condition = detection.get('class_name', '')
        confidence = detection.get('confidence_score', 0.0)
        self._cached_is_safe = (
            confidence >= 50.0 and 
            cloud_condition != 'Unknown' and 
            cloud_condition not in self._unsafe_conditions_set
        )
    
    def _setup_mqtt(self):
        """Setup and return MQTT client based on detect_config"""
        # Only setup if broker is configured
        if not self.detect_config.broker:
            logger.warning("MQTT broker not configured, MQTT publishing disabled")
            return None
            
        client = mqtt.Client()
        if self.detect_config.mqtt_username and self.detect_config.mqtt_password:
            client.username_pw_set(self.detect_config.mqtt_username, self.detect_config.mqtt_password)
        
        # Setup Last Will Testament for HA discovery mode
        if self.detect_config.mqtt_discovery_mode == 'homeassistant':
            availability_topic = f"{self.detect_config.mqtt_discovery_prefix}/sensor/clouddetect_{self.detect_config.device_id}/availability"
            client.will_set(availability_topic, "offline", retain=True)
            
        try:
            client.connect(self.detect_config.broker, self.detect_config.port)
            client.loop_start()
            logger.info(f"Connected to MQTT broker at {self.detect_config.broker}:{self.detect_config.port}")
            return client
        except Exception as e:
            logger.error(f"Failed to connect to MQTT broker: {e}")
            return None
    
    def get_next_transaction_id(self) -> int:
        """Generate next server transaction ID (thread-safe, wraps at uint32 max)"""
        with self.transaction_lock:
            self.server_transaction_id = (self.server_transaction_id + 1) % 4294967296
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
        """Background thread for continuous cloud detection AND MQTT publishing"""
        logger.info("Starting unified detection loop (ASCOM + MQTT)")
        
        while not self.stop_detection.is_set():
            try:
                # Perform detection in background without blocking API responses
                result = self.cloud_detector.detect()
                result['timestamp'] = datetime.now()
                
                # Update ASCOM state
                with self.detection_lock:
                    self.latest_detection = result
                    self._update_cached_safety(result)
                
                # Publish to MQTT (unified publishing)
                if self.mqtt_client:
                    try:
                        if self.detect_config.mqtt_discovery_mode == 'homeassistant':
                            self.ha_discovery.publish_states(result)
                        else:
                            # Legacy single-topic publishing
                            self.mqtt_client.publish(self.detect_config.topic, json.dumps(result))
                    except Exception as e:
                        logger.error(f"MQTT publish failed: {e}")
                
                logger.info(f"Cloud detection: {result['class_name']} "
                           f"({result['confidence_score']:.1f}%)")
            except Exception as e:
                logger.error(f"Error in detection loop: {e}")
                # On error, keep previous detection state
            
            # Wait for next update - check stop signal every second
            if self.stop_detection.wait(self.alpaca_config.update_interval):
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
            self.connected_at = datetime.now()
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
            self.connected_at = None
            logger.info("Disconnected from safety monitor")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def is_safe(self) -> bool:
        """Determine if conditions are safe based on latest detection"""
        with self.detection_lock:
            return self._cached_is_safe
    
    def get_device_state(self) -> list:
        """Get current operational state"""
        # Per ASCOM spec: DeviceState should only include operational properties
        # For SafetyMonitor, only IsSafe is operational
        with self.detection_lock:
            return [{"Name": "IsSafe", "Value": self._cached_is_safe if self.connected else False}]
        
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
discovery_service: Optional['AlpacaDiscovery'] = None


def validate_device_number(device_number: int) -> Optional[tuple]:
    """Validate device number and return error response if invalid"""
    if device_number != safety_monitor.alpaca_config.device_number:
        _, client_tx_id = safety_monitor.get_client_params()
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    return None


def create_simple_get_endpoint(attribute_getter):
    """Factory for simple GET endpoints that return a config value"""
    def endpoint(device_number: int):
        error_response = validate_device_number(device_number)
        if error_response:
            return error_response
        _, client_tx_id = safety_monitor.get_client_params()
        return jsonify(safety_monitor.create_response(
            value=attribute_getter(),
            client_transaction_id=client_tx_id
        ))
    return endpoint


@app.route('/api/v1/safetymonitor/<int:device_number>/issafe', methods=['GET'])
def get_issafe(device_number: int):
    """Get safety status"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    _, client_tx_id = safety_monitor.get_client_params()
    
    # Per ASCOM spec: Always return a value, never an error
    # If disconnected, return False (unsafe) to protect equipment
    is_safe = safety_monitor.is_safe() if safety_monitor.connected else False
    
    return jsonify(safety_monitor.create_response(
        value=is_safe,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/connected', methods=['GET'])
def get_connected(device_number: int):
    """Get connection state"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    _, client_tx_id = safety_monitor.get_client_params()
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.connected,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/connected', methods=['PUT'])
def set_connected(device_number: int):
    """Set connection state"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    _, client_tx_id = safety_monitor.get_client_params()
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
        if target_state != safety_monitor.connected:
            safety_monitor.connect() if target_state else safety_monitor.disconnect()
        
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
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    _, client_tx_id = safety_monitor.get_client_params()
    return jsonify(safety_monitor.create_response(
        value=safety_monitor.connecting,
        client_transaction_id=client_tx_id
    ))


@app.route('/api/v1/safetymonitor/<int:device_number>/connect', methods=['PUT'])
def connect_device(device_number: int):
    """Connect to device asynchronously (Platform 7, Interface V3+)"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    _, client_tx_id = safety_monitor.get_client_params()
    try:
        if not safety_monitor.connected:
            safety_monitor.connect()
        return jsonify(safety_monitor.create_response(client_transaction_id=client_tx_id))
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
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    _, client_tx_id = safety_monitor.get_client_params()
    try:
        if safety_monitor.connected:
            safety_monitor.disconnect()
        return jsonify(safety_monitor.create_response(client_transaction_id=client_tx_id))
    except Exception as e:
        logger.error(f"Error during disconnect: {e}")
        return jsonify(safety_monitor.create_response(
            error_number=ERROR_UNSPECIFIED,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500


@app.route('/api/v1/safetymonitor/<int:device_number>/description', methods=['GET'])
def get_description(device_number: int):
    return create_simple_get_endpoint(lambda: safety_monitor.alpaca_config.device_description)(device_number)


@app.route('/api/v1/safetymonitor/<int:device_number>/devicestate', methods=['GET'])
def get_devicestate(device_number: int):
    return create_simple_get_endpoint(safety_monitor.get_device_state)(device_number)


@app.route('/api/v1/safetymonitor/<int:device_number>/driverinfo', methods=['GET'])
def get_driverinfo(device_number: int):
    return create_simple_get_endpoint(lambda: safety_monitor.alpaca_config.driver_info)(device_number)


@app.route('/api/v1/safetymonitor/<int:device_number>/driverversion', methods=['GET'])
def get_driverversion(device_number: int):
    return create_simple_get_endpoint(lambda: safety_monitor.alpaca_config.driver_version)(device_number)


@app.route('/api/v1/safetymonitor/<int:device_number>/interfaceversion', methods=['GET'])
def get_interfaceversion(device_number: int):
    return create_simple_get_endpoint(lambda: safety_monitor.alpaca_config.interface_version)(device_number)


@app.route('/api/v1/safetymonitor/<int:device_number>/name', methods=['GET'])
def get_name(device_number: int):
    return create_simple_get_endpoint(lambda: safety_monitor.alpaca_config.device_name)(device_number)


@app.route('/api/v1/safetymonitor/<int:device_number>/supportedactions', methods=['GET'])
def get_supportedactions(device_number: int):
    return create_simple_get_endpoint(lambda: [])(device_number)


@app.route('/api/v1/safetymonitor/<int:device_number>/action', methods=['PUT'])
def put_action(device_number: int):
    """Execute custom action (not implemented)"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    _, client_tx_id = safety_monitor.get_client_params()
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
        "ManufacturerVersion": safety_monitor.alpaca_config.driver_version,
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
    
    # Get available conditions from labels.txt - used by both GET and POST
    all_available_conditions = get_available_cloud_conditions()
    
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
        for condition in all_available_conditions:
            if request.form.get(f'unsafe_{condition}'):
                unsafe_conditions.append(condition)
        
        safety_monitor.alpaca_config.unsafe_conditions = unsafe_conditions
        # Update the in-memory set to match the new configuration
        safety_monitor._unsafe_conditions_set = set(unsafe_conditions)
        
        # Recalculate cached safety status with new unsafe conditions
        with safety_monitor.detection_lock:
            safety_monitor._update_cached_safety(safety_monitor.latest_detection)
        
        logger.info(f"Unsafe conditions updated to: {unsafe_conditions}")
        
        # Save configuration to file
        safety_monitor.alpaca_config.save_to_file()
        
        # Redirect to prevent form resubmission (Post/Redirect/Get pattern)
        return redirect(url_for('setup_device', device_number=device_number))
    
    # GET request - show the form
    message = ""
    
    # HTML form for setup
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>SimpleCloudDetect Setup - Command Center</title>
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap" rel="stylesheet">
        <style>
            /* Typography Imports */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

            /* Custom Scrollbar */
            ::-webkit-scrollbar { width: 6px; }
            ::-webkit-scrollbar-track { background: rgba(15, 23, 42, 0.2); }
            ::-webkit-scrollbar-thumb { background: rgba(51, 65, 85, 0.5); border-radius: 4px; }
            ::-webkit-scrollbar-thumb:hover { background: rgba(71, 85, 105, 0.8); }

            /* Glassmorphism Utility */
            .glass-panel {
                background: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
            }

            /* Input Glow Effect */
            .glow-input:focus {
                box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
            }

            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: rgb(2, 6, 23);
                min-height: 100vh;
                padding: 20px;
                color: rgb(226, 232, 240);
                position: relative;
            }
            
            /* Background Overlay */
            body::before {
                content: '';
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(to right, rgba(2, 6, 23, 0.95), rgba(2, 6, 23, 0.4));
                pointer-events: none;
                z-index: 0;
            }
            
            .container {
                max-width: 1400px;
                margin: 50px auto;
                position: relative;
                z-index: 1;
            }
            
            .header {
                background: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 32px 30px;
                text-align: left;
                position: relative;
                margin-bottom: 24px;
                box-shadow: 0 0 20px rgba(8, 145, 178, 0.1);
            }
            
            .github-btn {
                position: absolute;
                top: 20px;
                right: 20px;
                background: rgba(6, 182, 212, 0.15);
                color: rgb(255, 255, 255);
                padding: 10px 20px;
                border-radius: 8px;
                text-decoration: none;
                font-size: 14px;
                font-weight: 600;
                transition: all 0.3s ease;
                border: 2px solid rgba(6, 182, 212, 0.6);
                display: inline-flex;
                align-items: center;
                gap: 8px;
            }
            
            .github-btn:hover {
                background: rgba(6, 182, 212, 0.25);
                border-color: rgb(6, 182, 212);
                transform: translateY(-2px);
                color: rgb(34, 211, 238);
            }
            
            h1 {
                color: #ffffff;
                font-size: 32px;
                font-weight: 700;
                letter-spacing: -0.5px;
                margin: 0;
                text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
            }
            
            h1 .highlight {
                color: rgb(34, 211, 238);
            }
            
            .two-column-layout {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 24px;
                align-items: start;
            }
            
            @media (max-width: 1024px) {
                .two-column-layout {
                    grid-template-columns: 1fr;
                }
            }
            
            .left-column,
            .right-column {
                background: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 30px;
                box-shadow: 0 0 20px rgba(8, 145, 178, 0.1);
            }
            
            .column-header {
                color: #ffffff;
                font-size: 24px;
                font-weight: 700;
                letter-spacing: -0.5px;
                margin: 0 0 24px 0;
                text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
                border-bottom: 2px solid rgba(6, 182, 212, 0.3);
                padding-bottom: 12px;
            }
            
            .column-header .highlight {
                color: rgb(34, 211, 238);
            }
            
            h2 {
                color: rgb(34, 211, 238);
                margin-top: 30px;
                margin-bottom: 15px;
                font-size: 20px;
                font-weight: 600;
                letter-spacing: -0.25px;
                text-transform: uppercase;
                font-size: 14px;
                border-bottom: 1px solid rgba(6, 182, 212, 0.3);
                padding-bottom: 8px;
            }
            
            .form-group {
                margin-bottom: 24px;
            }
            
            label {
                display: block;
                margin-bottom: 8px;
                color: rgb(148, 163, 184);
                font-weight: 500;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            input[type="text"] {
                width: 100%;
                padding: 12px 14px;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgb(71, 85, 105);
                border-radius: 6px;
                color: rgb(241, 245, 249);
                font-size: 14px;
                font-family: 'Inter', sans-serif;
                transition: all 0.3s ease;
            }
            
            input[type="text"]:focus {
                outline: none;
                border-color: rgb(6, 182, 212);
                background: rgba(15, 23, 42, 0.8);
                box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
            }
            
            input[type="text"]::placeholder {
                color: rgb(100, 116, 139);
            }
            
            .checkbox-group {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
                gap: 12px;
                padding: 20px;
                background: rgba(10, 15, 24, 0.9);
                border-radius: 8px;
                border: 1px solid rgba(71, 85, 105, 0.5);
            }
            
            .checkbox-item {
                display: flex;
                align-items: center;
                padding: 12px 14px;
                background: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(12px);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                transition: all 0.3s ease;
            }
            
            .checkbox-item:hover {
                background: rgba(6, 182, 212, 0.1);
                border-color: rgba(6, 182, 212, 0.5);
                box-shadow: 0 0 15px rgba(6, 182, 212, 0.2);
            }
            
            .checkbox-item input[type="checkbox"] {
                margin-right: 12px;
                width: 18px;
                height: 18px;
                cursor: pointer;
                accent-color: rgb(6, 182, 212);
            }
            
            .checkbox-item label {
                margin: 0;
                font-weight: 400;
                cursor: pointer;
                color: rgb(226, 232, 240);
                font-size: 14px;
                text-transform: none;
                letter-spacing: 0;
            }
            
            button {
                background: rgba(6, 182, 212, 0.7);
                color: rgb(226, 232, 240);
                padding: 12px 24px;
                border: 1px solid rgba(6, 182, 212, 0.5);
                border-radius: 6px;
                cursor: pointer;
                font-size: 14px;
                font-weight: 600;
                width: 100%;
                margin-top: 24px;
                transition: all 0.3s ease;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }
            
            button:hover {
                background: rgba(6, 182, 212, 0.85);
                border-color: rgb(6, 182, 212);
                transform: translateY(-1px);
            }
            
            button:active {
                transform: scale(0.98);
            }
            
            .message {
                background: rgba(16, 185, 129, 0.1);
                color: rgb(52, 211, 153);
                padding: 14px 16px;
                border-radius: 6px;
                margin-bottom: 20px;
                font-weight: 500;
                font-size: 14px;
                border: 1px solid rgba(16, 185, 129, 0.3);
                animation: slideIn 0.3s ease;
            }
            
            @keyframes slideIn {
                from {
                    opacity: 0;
                    transform: translateY(-10px);
                }
                to {
                    opacity: 1;
                    transform: translateY(0);
                }
            }
            
            .info {
                background: rgba(10, 15, 24, 0.9);
                padding: 18px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid rgba(71, 85, 105, 0.5);
            }
            
            .info p {
                margin: 8px 0;
                color: rgb(148, 163, 184);
                line-height: 1.8;
                font-size: 14px;
                font-family: 'JetBrains Mono', monospace;
            }
            
            .info strong {
                color: rgb(226, 232, 240);
                font-weight: 600;
            }
            
            .info-title {
                color: rgb(34, 211, 238);
                font-weight: 600;
                font-size: 16px;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 1px;
                border-bottom: 1px solid rgba(6, 182, 212, 0.3);
                padding-bottom: 6px;
            }
            
            .help-text {
                font-size: 12px;
                color: rgb(100, 116, 139);
                margin-top: 6px;
                margin-bottom: 12px;
            }
            
            .safe-indicator {
                color: rgb(52, 211, 153);
                font-weight: 600;
            }
            
            .unsafe-indicator {
                color: rgb(248, 113, 113);
                font-weight: 600;
            }
            
            .status-connected {
                color: rgb(52, 211, 153);
                font-weight: 700;
                text-transform: uppercase;
            }
            
            .status-disconnected {
                color: rgb(248, 113, 113);
                font-weight: 700;
                text-transform: uppercase;
            }
            
            /* Terminal/Console Styling */
            .terminal-line {
                color: rgb(100, 116, 139);
                font-family: 'JetBrains Mono', monospace;
            }
            
            .terminal-line .prompt {
                color: rgb(34, 211, 238);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <a href="https://github.com/chvvkumar/simpleCloudDetect" target="_blank" class="github-btn">
                    GitHub
                    <svg width="16" height="16" fill="currentColor" viewBox="0 0 16 16">
                        <path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8z"/>
                    </svg>
                </a>
                <h1>☁️ Simple<span class="highlight">Cloud</span>Detect</h1>
            </div>
            
            <div class="two-column-layout">
                <!-- Left Column: Information Display -->
                <div class="left-column">
                    {% if message %}
                    <div class="message">✓ {{ message }}</div>
                    {% endif %}
                    
                    <div class="info">
                        <div class="info-title">⚡ Detection Status</div>
                        <p>Condition: <strong>{{ current_condition }}</strong></p>
                        <p>Confidence: <strong>{{ current_confidence }}%</strong></p>
                        <p>Detection Time: <strong>{{ detection_time }}s</strong></p>
                        <p>Last Updated: <strong>{{ last_update }}</strong></p>
                    </div>
                    
                    <div class="info">
                        <div class="info-title">🔌 ASCOM Connection</div>
                        <p>Status: <span class="{{ ascom_status_class }}">{{ ascom_status }}</span></p>
                        {% if connection_duration %}
                        <p>Duration: <strong>{{ connection_duration }}</strong></p>
                        {% endif %}
                    </div>
                    
                    <div class="info">
                        <div class="info-title">⚙️ Current Configuration</div>
                        <p>Device Name: <strong>{{ current_name }}</strong></p>
                        <p>Location: <strong>{{ current_location }}</strong></p>
                        <p>Safe Conditions: <span class="safe-indicator">{{ safe_conditions|join(', ') }}</span></p>
                        <p>Unsafe Conditions: <span class="unsafe-indicator">{{ unsafe_conditions|join(', ') }}</span></p>
                    </div>
                </div>
                
                <!-- Right Column: User Input -->
                <div class="right-column">
                    <h2 class="column-header"><span class="highlight">Setup</span></h2>
                    
                    <form method="POST">
                        <div class="form-group">
                            <label for="device_name">Device Name</label>
                            <input type="text" id="device_name" name="device_name" class="glow-input"
                                   value="{{ current_name }}" placeholder="Enter device name">
                        </div>
                        
                        <div class="form-group">
                            <label for="location">Location</label>
                            <input type="text" id="location" name="location" class="glow-input"
                                   value="{{ current_location }}" placeholder="Enter location">
                        </div>
                        
                        <div class="form-group">
                            <h2>🛡️ Safety Configuration</h2>
                            <label>Mark conditions that are UNSAFE for observing</label>
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
            </div>
        </div>
    </body>
    </html>
    """
    
    # Calculate safe vs unsafe conditions for display
    unsafe_conditions = safety_monitor.alpaca_config.unsafe_conditions
    safe_conditions = [c for c in all_available_conditions if c not in unsafe_conditions]
    
    # Get current detection data
    with safety_monitor.detection_lock:
        current_condition = safety_monitor.latest_detection.get('class_name', 'Unknown')
        current_confidence = round(safety_monitor.latest_detection.get('confidence_score', 0.0), 1)
        detection_time = round(safety_monitor.latest_detection.get('Detection Time (Seconds)', 0.0), 2)
        timestamp = safety_monitor.latest_detection.get('timestamp')
        
    # Format timestamp
    if timestamp:
        last_update = timestamp.strftime('%Y-%m-%d %H:%M:%S')
    else:
        last_update = 'Never'
    
    # Check ASCOM connection status
    if safety_monitor.connected:
        ascom_status = 'Connected'
        ascom_status_class = 'status-connected'
        
        # Calculate connection duration
        if safety_monitor.connected_at:
            duration = datetime.now() - safety_monitor.connected_at
            hours, remainder = divmod(int(duration.total_seconds()), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                connection_duration = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                connection_duration = f"{minutes}m {seconds}s"
            else:
                connection_duration = f"{seconds}s"
        else:
            connection_duration = None
    else:
        ascom_status = 'Disconnected'
        ascom_status_class = 'status-disconnected'
        connection_duration = None
    
    return render_template_string(
        html_template,
        message=message,
        current_name=safety_monitor.alpaca_config.device_name,
        current_location=safety_monitor.alpaca_config.location,
        all_conditions=all_available_conditions,
        unsafe_conditions=unsafe_conditions,
        safe_conditions=safe_conditions,
        current_condition=current_condition,
        current_confidence=current_confidence,
        detection_time=detection_time,
        last_update=last_update,
        ascom_status=ascom_status,
        ascom_status_class=ascom_status_class,
        connection_duration=connection_duration
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
def main():
    """Main entry point for Waitress-based unified service"""
    from waitress import serve
    
    global safety_monitor, discovery_service
    
    # 1. Load Configurations
    detect_config = DetectConfig.from_env()
    alpaca_config = AlpacaConfig.load_from_file()
    
    if alpaca_config is None:
        # Create default config from environment
        alpaca_config = AlpacaConfig(
            port=int(os.getenv('ALPACA_PORT', '11111')),
            device_number=int(os.getenv('ALPACA_DEVICE_NUMBER', '0')),
            update_interval=int(os.getenv('ALPACA_UPDATE_INTERVAL', '30'))
        )
        alpaca_config.save_to_file()
    else:
        # Override port settings from environment if provided
        alpaca_config.port = int(os.getenv('ALPACA_PORT', str(alpaca_config.port)))
        alpaca_config.device_number = int(os.getenv('ALPACA_DEVICE_NUMBER', str(alpaca_config.device_number)))
        alpaca_config.update_interval = int(os.getenv('ALPACA_UPDATE_INTERVAL', str(alpaca_config.update_interval)))
    
    # 2. Initialize global safety monitor
    safety_monitor = AlpacaSafetyMonitor(alpaca_config, detect_config)
    
    # 3. Connect (starts the background detection thread)
    safety_monitor.connect()
    
    # 4. Start Discovery Service
    discovery_service = AlpacaDiscovery(alpaca_config.port)
    discovery_service.start()
    
    logger.info(f"ASCOM Alpaca SafetyMonitor initialized")
    logger.info(f"Device: {alpaca_config.device_name}")
    logger.info(f"Update interval: {alpaca_config.update_interval}s")
    logger.info(f"Unsafe conditions: {', '.join(alpaca_config.unsafe_conditions)}")
    logger.info(f"Safe conditions: {', '.join([c for c in ALL_CLOUD_CONDITIONS if c not in alpaca_config.unsafe_conditions])}")
    
    # 5. Setup signal handlers for graceful shutdown
    def handle_shutdown_signal(signum, frame):
        logger.info(f"Received shutdown signal ({signum})")
        try:
            discovery_service.stop()
            safety_monitor.disconnect()
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
        finally:
            sys.exit(0)
    
    signal.signal(signal.SIGTERM, handle_shutdown_signal)
    signal.signal(signal.SIGINT, handle_shutdown_signal)
    
    # 6. Run Waitress Server (production-ready, single-process, multi-threaded)
    logger.info(f"Starting Waitress Server on 0.0.0.0:{alpaca_config.port}")
    logger.info("Press Ctrl+C to stop the server")
    
    try:
        serve(app, host='0.0.0.0', port=alpaca_config.port, threads=6)
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received")
        handle_shutdown_signal(signal.SIGINT, None)


if __name__ == '__main__':
    main()
