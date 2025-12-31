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
from zoneinfo import ZoneInfo
from typing import Optional, Dict, Any
import json
import base64
import io

from flask import Flask, request, jsonify, render_template_string, redirect, url_for, Response
from flask_cors import CORS
import paho.mqtt.client as mqtt
from PIL import Image

from detect import CloudDetector, Config as DetectConfig, HADiscoveryManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
start_time = datetime.now()

# Global timezone helper
def get_current_time(timezone_str: str = 'UTC') -> datetime:
    """Get current time in specified timezone"""
    try:
        tz = ZoneInfo(timezone_str)
        return datetime.now(tz)
    except Exception:
        # Fallback to UTC if timezone is invalid
        return datetime.now(ZoneInfo('UTC'))

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
    detection_interval: int = 30  # seconds between ML detections (from detect.py)
    update_interval: int = 30  # seconds between cloud detection updates
    location: str = "AllSky Camera"
    image_url: str = field(default_factory=lambda: os.environ.get('IMAGE_URL', ''))
    unsafe_conditions: list = field(default_factory=lambda: ['Rain', 'Snow', 'Mostly Cloudy', 'Overcast'])
    
    # Confidence threshold settings
    default_threshold: float = 50.0  # Default threshold for any class not explicitly configured
    class_thresholds: Dict[str, float] = field(default_factory=dict)  # Map class names to thresholds
    
    # Debounce settings (in seconds)
    debounce_to_safe_sec: int = 60  # Wait time before switching from Unsafe → Safe
    debounce_to_unsafe_sec: int = 0  # Wait time before switching from Safe → Unsafe (immediate)
    
    # NTP and timezone settings
    ntp_server: str = field(default_factory=lambda: os.environ.get('NTP_SERVER', 'pool.ntp.org'))
    timezone: str = field(default_factory=lambda: os.environ.get('TZ', 'UTC'))
    
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
        """Load configuration from JSON file with backward compatibility"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)
                
                # Ensure new fields exist with defaults for backward compatibility
                if 'default_threshold' not in config_dict:
                    config_dict['default_threshold'] = 50.0
                if 'class_thresholds' not in config_dict:
                    config_dict['class_thresholds'] = {}
                if 'debounce_to_safe_sec' not in config_dict:
                    config_dict['debounce_to_safe_sec'] = 60
                if 'debounce_to_unsafe_sec' not in config_dict:
                    config_dict['debounce_to_unsafe_sec'] = 0
                if 'detection_interval' not in config_dict:
                    config_dict['detection_interval'] = 30
                if 'image_url' not in config_dict:
                    config_dict['image_url'] = os.environ.get('IMAGE_URL', '')
                
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
        self.disconnected_at: Optional[datetime] = None
        self.last_connected_at: Optional[datetime] = None  # Track last successful connection
        self.client_ip: Optional[str] = None  # Track connected client's IP address
        self.last_session_duration: Optional[float] = None
        
        # Cloud detection state - use single lock for all state
        self.latest_detection: Optional[Dict[str, Any]] = {
            'class_name': 'Unknown',
            'confidence_score': 0.0,
            'Detection Time (Seconds)': 0.0,
            'timestamp': None
        }
        self._cached_is_safe = False  # Cache safe status to reduce lock contention
        self._unsafe_conditions_set = set(alpaca_config.unsafe_conditions)  # Set lookup is O(1)
        
        # Debounce state tracking
        self._stable_safe_state = False  # The actual safe/unsafe value returned to the client
        self._pending_safe_state: Optional[bool] = None  # The potential new state we are waiting to verify
        self._state_change_start_time: Optional[datetime] = None  # Timestamp when pending state first appeared
        
        # Safety state history (last 100 transitions)
        self._safety_history: list = []  # List of (timestamp, is_safe, condition, confidence) tuples
        
        # Latest detection image (base64 encoded)
        self.latest_image_data: Optional[str] = None
        
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
            initial_result = self.cloud_detector.detect(return_image=True)
            initial_result['timestamp'] = get_current_time(self.alpaca_config.timezone)
            
            # Create thumbnail from the image that was just used for detection
            initial_image = None
            if 'image' in initial_result:
                initial_image = self._create_thumbnail_from_image(initial_result['image'])
                del initial_result['image']
            
            with self.detection_lock:
                self.latest_detection = initial_result
                self.latest_image_data = initial_image
                self._update_cached_safety(initial_result)
                
                # Add initial state to safety history
                self._safety_history.append({
                    'timestamp': get_current_time(self.alpaca_config.timezone),
                    'is_safe': self._stable_safe_state,
                    'condition': initial_result.get('class_name', 'Unknown'),
                    'confidence': initial_result.get('confidence_score', 0.0)
                })
                
            logger.info(f"Initial detection complete: {initial_result['class_name']}")
        except Exception as e:
            logger.error(f"Initial detection failed: {e}")
        
        logger.info(f"Initialized {self.alpaca_config.device_name}")
    
    def _create_thumbnail_from_image(self, img: Image.Image) -> Optional[str]:
        """Create base64-encoded thumbnail from a PIL image"""
        try:
            # Resize to thumbnail size (200x200) to reduce data size
            img_copy = img.copy()
            img_copy.thumbnail((200, 200), Image.Resampling.LANCZOS)
            
            # Convert to base64
            buffered = io.BytesIO()
            img_copy.save(buffered, format="JPEG", quality=85)
            img_str = base64.b64encode(buffered.getvalue()).decode()
            
            return img_str
        except Exception as e:
            logger.warning(f"Failed to create thumbnail: {e}")
            return None
    
    def _update_cached_safety(self, detection: Dict[str, Any]):
        """Update cached safety status with debouncing logic (assumes lock is held)"""
        # Step A: Determine Instantaneous Safety
        class_name = detection.get('class_name', '')
        confidence = detection.get('confidence_score', 0.0)
        
        # Get the specific threshold for this class, or use default
        threshold = self.alpaca_config.class_thresholds.get(
            class_name, 
            self.alpaca_config.default_threshold
        )
        
        # Check if current conditions indicate safe state
        is_safe_now = (
            confidence >= threshold and 
            class_name != 'Unknown' and 
            class_name not in self._unsafe_conditions_set
        )
        
        # Step B: Apply Debouncing
        if is_safe_now == self._stable_safe_state:
            # State matches - reset any pending changes
            self._pending_safe_state = None
            self._state_change_start_time = None
            self._cached_is_safe = self._stable_safe_state  # Keep cache in sync
        else:
            # State differs from stable state
            if self._pending_safe_state != is_safe_now:
                # New change detected - start debounce timer
                self._pending_safe_state = is_safe_now
                self._state_change_start_time = get_current_time(self.alpaca_config.timezone)
                logger.info(f"State change detected: {'Safe' if is_safe_now else 'Unsafe'} "
                           f"(pending debounce verification)")
            else:
                # Change persisting - check if debounce period has elapsed
                if self._state_change_start_time:
                    elapsed_time = (get_current_time(self.alpaca_config.timezone) - self._state_change_start_time).total_seconds()
                    
                    # Determine required duration based on transition direction
                    if is_safe_now:
                        # Transitioning to Safe - use safe debounce time
                        required_duration = self.alpaca_config.debounce_to_safe_sec
                    else:
                        # Transitioning to Unsafe - use unsafe debounce time (usually 0 for immediate)
                        required_duration = self.alpaca_config.debounce_to_unsafe_sec
                    
                    if elapsed_time >= required_duration:
                        # Debounce period complete - commit state change
                        self._stable_safe_state = is_safe_now
                        self._cached_is_safe = is_safe_now
                        self._pending_safe_state = None
                        self._state_change_start_time = None
                        
                        # Add to safety history (keep last 100)
                        self._safety_history.append({
                            'timestamp': get_current_time(self.alpaca_config.timezone),
                            'is_safe': is_safe_now,
                            'condition': class_name,
                            'confidence': confidence
                        })
                        if len(self._safety_history) > 100:
                            self._safety_history.pop(0)
                        
                        logger.warning(f"SAFETY STATE CHANGED: {'SAFE' if is_safe_now else 'UNSAFE'} "
                                     f"(class={class_name}, confidence={confidence:.1f}%, "
                                     f"threshold={threshold:.1f}%, debounce={elapsed_time:.1f}s)")
                    else:
                        # Still waiting - log progress
                        remaining = required_duration - elapsed_time
                        logger.debug(f"Debouncing: {remaining:.1f}s remaining for state change to "
                                   f"{'Safe' if is_safe_now else 'Unsafe'}")
    
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
                # Request the image to be returned so we can create a thumbnail
                result = self.cloud_detector.detect(return_image=True)
                result['timestamp'] = get_current_time(self.alpaca_config.timezone)
                
                # Create thumbnail from the image that was just used for detection
                image_data = None
                if 'image' in result:
                    image_data = self._create_thumbnail_from_image(result['image'])
                    # Remove image from result before storing (don't need it in detection dict)
                    del result['image']
                
                # Update ASCOM state
                with self.detection_lock:
                    self.latest_detection = result
                    self.latest_image_data = image_data
                    self._update_cached_safety(result)
                
                # Publish to MQTT (unified publishing)
                if self.mqtt_client:
                    try:
                        # Convert datetime to ISO format for JSON serialization
                        mqtt_result = result.copy()
                        if 'timestamp' in mqtt_result and isinstance(mqtt_result['timestamp'], datetime):
                            mqtt_result['timestamp'] = mqtt_result['timestamp'].isoformat()
                        
                        if self.detect_config.mqtt_discovery_mode == 'homeassistant':
                            self.ha_discovery.publish_states(mqtt_result)
                        else:
                            # Legacy single-topic publishing
                            self.mqtt_client.publish(self.detect_config.topic, json.dumps(mqtt_result))
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
    
    def connect(self, client_ip: Optional[str] = None):
        """Connect to the device (ASCOM client connection)"""
        if self.connected:
            return
        
        try:
            # Just mark as connected - detection loop is already running
            self.connected = True
            self.connected_at = get_current_time(self.alpaca_config.timezone)
            self.last_connected_at = self.connected_at
            self.client_ip = client_ip
            logger.info(f"ASCOM client connected to safety monitor from {client_ip or 'unknown'}")
        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from the device (ASCOM client disconnection)"""
        if not self.connected:
            return
        
        try:
            # Just mark as disconnected - detection loop keeps running for MQTT/web UI
            self.connected = False
            self.disconnected_at = get_current_time(self.alpaca_config.timezone)
            if self.connected_at:
                duration = (self.disconnected_at - self.connected_at).total_seconds()
                self.last_session_duration = duration
            self.connected_at = None
            logger.info("ASCOM client disconnected from safety monitor")
        except Exception as e:
            logger.error(f"Error during disconnect: {e}")
    
    def is_safe(self) -> bool:
        """Determine if conditions are safe based on latest detection"""
        with self.detection_lock:
            return self._stable_safe_state
    
    def get_device_state(self) -> list:
        """Get current operational state"""
        # Per ASCOM spec: DeviceState should only include operational properties
        # For SafetyMonitor, only IsSafe is operational
        with self.detection_lock:
            return [{"Name": "IsSafe", "Value": self._stable_safe_state if self.connected else False}]
        
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
            if target_state:
                client_ip = request.remote_addr
                safety_monitor.connect(client_ip=client_ip)
            else:
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
            client_ip = request.remote_addr
            safety_monitor.connect(client_ip=client_ip)
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


@app.route('/api/v1/latest_image', methods=['GET'])
def get_latest_image():
    """Serve the latest detection image as JPEG"""
    with safety_monitor.detection_lock:
        image_data = safety_monitor.latest_image_data
    
    if image_data:
        try:
            # Decode base64 and return as image
            img_bytes = base64.b64decode(image_data)
            return Response(img_bytes, mimetype='image/jpeg')
        except Exception as e:
            logger.error(f"Failed to serve image: {e}")
            return Response(status=404)
    else:
        return Response(status=404)


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


def sync_class_thresholds_with_labels():
    """Synchronize class_thresholds with available labels, adding defaults for new labels"""
    if safety_monitor:
        all_conditions = get_available_cloud_conditions()
        current_thresholds = safety_monitor.alpaca_config.class_thresholds
        
        # Check if any new labels were added that don't have thresholds
        for condition in all_conditions:
            if condition not in current_thresholds:
                # Initialize new conditions with default threshold
                current_thresholds[condition] = safety_monitor.alpaca_config.default_threshold
                logger.info(f"Initialized threshold for new condition '{condition}': {safety_monitor.alpaca_config.default_threshold}%")
        
        # Optionally remove thresholds for conditions no longer in labels
        removed_conditions = [cond for cond in current_thresholds if cond not in all_conditions]
        for condition in removed_conditions:
            del current_thresholds[condition]
            logger.info(f"Removed threshold for obsolete condition '{condition}'")
        
        if removed_conditions:
            # Save if we made changes
            safety_monitor.alpaca_config.save_to_file()


@app.route('/setup/v1/safetymonitor/<int:device_number>/setup', methods=['GET', 'POST'])
def setup_device(device_number: int):
    """Setup page for configuring device name and location"""
    if device_number != safety_monitor.alpaca_config.device_number:
        return jsonify({"error": "Invalid device number"}), 404
    
    # Synchronize thresholds with available labels (handles new labels in labels.txt)
    sync_class_thresholds_with_labels()
    
    # Get available conditions from labels.txt - used by both GET and POST
    all_available_conditions = get_available_cloud_conditions()
    
    if request.method == 'POST':
        # Handle form submission
        device_name = request.form.get('device_name', '').strip()
        location = request.form.get('location', '').strip()
        image_url = request.form.get('image_url', '').strip()
        
        if device_name:
            safety_monitor.alpaca_config.device_name = device_name
            logger.info(f"Device name updated to: {device_name}")
        
        if location:
            safety_monitor.alpaca_config.location = location
            logger.info(f"Location updated to: {location}")
        
        if image_url:
            safety_monitor.alpaca_config.image_url = image_url
            # Update the detect_config as well
            safety_monitor.detect_config.image_url = image_url
            logger.info(f"Image URL updated to: {image_url}")
        elif image_url == '' and 'image_url' in request.form:
            # User explicitly cleared the field - use environment default
            default_url = os.environ.get('IMAGE_URL', '')
            safety_monitor.alpaca_config.image_url = default_url
            safety_monitor.detect_config.image_url = default_url
            logger.info(f"Image URL reset to environment default: {default_url}")
        
        # Handle NTP and timezone settings
        ntp_server = request.form.get('ntp_server', '').strip()
        if ntp_server:
            safety_monitor.alpaca_config.ntp_server = ntp_server
            logger.info(f"NTP server updated to: {ntp_server}")
        
        timezone = request.form.get('timezone', '').strip()
        if timezone:
            safety_monitor.alpaca_config.timezone = timezone
            logger.info(f"Timezone updated to: {timezone}")
        
        # Handle timing configuration with safety validations
        try:
            detection_interval = request.form.get('detection_interval', '').strip()
            if detection_interval:
                val = int(detection_interval)
                if 5 <= val <= 300:
                    safety_monitor.alpaca_config.detection_interval = val
                    # Update the detect_config as well
                    safety_monitor.detect_config.detect_interval = val
                    logger.info(f"Detection interval updated to: {val}s")
                else:
                    logger.warning(f"Detection interval {val}s out of range (5-300s)")
        except ValueError:
            logger.warning(f"Invalid detection_interval value: {detection_interval}")
        
        try:
            update_interval = request.form.get('update_interval', '').strip()
            if update_interval:
                val = int(update_interval)
                if 5 <= val <= 300:
                    safety_monitor.alpaca_config.update_interval = val
                    logger.info(f"ASCOM update interval updated to: {val}s")
                else:
                    logger.warning(f"Update interval {val}s out of range (5-300s)")
        except ValueError:
            logger.warning(f"Invalid update_interval value: {update_interval}")
        
        # Handle debounce timers with safety validations
        try:
            debounce_safe = request.form.get('debounce_safe', '').strip()
            if debounce_safe:
                val = int(debounce_safe)
                # Validation: Safe debounce should be >= detection_interval for smooth operation
                if val > 0 and val < safety_monitor.alpaca_config.detection_interval:
                    logger.warning(f"Safe wait time {val}s < detection interval {safety_monitor.alpaca_config.detection_interval}s - may cause erratic behavior")
                safety_monitor.alpaca_config.debounce_to_safe_sec = val
                logger.info(f"Debounce to safe updated to: {val}s")
        except ValueError:
            logger.warning(f"Invalid debounce_safe value: {debounce_safe}")
        
        try:
            debounce_unsafe = request.form.get('debounce_unsafe', '').strip()
            if debounce_unsafe:
                val = int(debounce_unsafe)
                # Safety validation: Warn if unsafe debounce > 30s (delays emergency response)
                if val > 30:
                    logger.warning(f"Unsafe wait time {val}s > 30s - delays emergency response to dangerous conditions")
                safety_monitor.alpaca_config.debounce_to_unsafe_sec = val
                logger.info(f"Debounce to unsafe updated to: {val}s")
        except ValueError:
            logger.warning(f"Invalid debounce_unsafe value: {debounce_unsafe}")
        
        # Handle unsafe conditions from radio buttons
        unsafe_conditions = []
        for condition in all_available_conditions:
            safety_choice = request.form.get(f'safety_{condition}')
            if safety_choice == 'unsafe':
                unsafe_conditions.append(condition)
        
        safety_monitor.alpaca_config.unsafe_conditions = unsafe_conditions
        # Update the in-memory set to match the new configuration
        safety_monitor._unsafe_conditions_set = set(unsafe_conditions)
        
        # Handle class-specific thresholds
        class_thresholds = {}
        for condition in all_available_conditions:
            threshold_value = request.form.get(f'threshold_{condition}', '').strip()
            if threshold_value:
                try:
                    threshold = float(threshold_value)
                    if 0 <= threshold <= 100:
                        class_thresholds[condition] = threshold
                    else:
                        logger.warning(f"Threshold for {condition} out of range: {threshold}")
                except ValueError:
                    logger.warning(f"Invalid threshold for {condition}: {threshold_value}")
        
        safety_monitor.alpaca_config.class_thresholds = class_thresholds
        logger.info(f"Class thresholds updated: {class_thresholds}")
        
        # Recalculate cached safety status with new configuration
        with safety_monitor.detection_lock:
            # Reset debounce state when configuration changes
            safety_monitor._pending_safe_state = None
            safety_monitor._state_change_start_time = None
            safety_monitor._update_cached_safety(safety_monitor.latest_detection)
        
        logger.info(f"Unsafe conditions updated to: {unsafe_conditions}")
        
        # Save configuration to file
        safety_monitor.alpaca_config.save_to_file()
        
        # Trigger immediate detection with new settings
        logger.info("Triggering immediate detection after config save")
        try:
            result = safety_monitor.cloud_detector.detect(return_image=True)
            result['timestamp'] = get_current_time(safety_monitor.alpaca_config.timezone)
            
            # Create thumbnail from the image that was just used for detection
            image_data = None
            if 'image' in result:
                image_data = safety_monitor._create_thumbnail_from_image(result['image'])
                del result['image']
            
            with safety_monitor.detection_lock:
                safety_monitor.latest_detection = result
                safety_monitor.latest_image_data = image_data
                safety_monitor._update_cached_safety(result)
            logger.info(f"Immediate detection after config save: {result['class_name']} ({result['confidence_score']:.1f}%)")
        except Exception as e:
            logger.error(f"Immediate detection after config save failed: {e}")
        
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
            
            .main-content {
                background: rgba(15, 23, 42, 0.6);
                backdrop-filter: blur(12px);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 12px;
                padding: 32px;
                box-shadow: 0 0 20px rgba(8, 145, 178, 0.1);
                margin-bottom: 24px;
            }
            
            .section {
                margin-bottom: 32px;
            }
            
            .section:last-child {
                margin-bottom: 0;
            }
            
            .section-header {
                color: #ffffff;
                font-size: 20px;
                font-weight: 700;
                letter-spacing: -0.5px;
                margin: 0 0 16px 0;
                text-shadow: 0 0 20px rgba(6, 182, 212, 0.5);
                border-bottom: 2px solid rgba(6, 182, 212, 0.3);
                padding-bottom: 10px;
                display: flex;
                align-items: center;
                gap: 10px;
            }
            
            .section-header .icon {
                font-size: 24px;
            }
            
            .section-header .highlight {
                color: rgb(34, 211, 238);
            }
            
            .status-grid {
                display: flex;
                flex-direction: column;
                gap: 16px;
                margin-bottom: 24px;
            }
            
            .status-card {
                background: rgba(10, 15, 24, 0.9);
                padding: 16px;
                border-radius: 8px;
                border: 1px solid rgba(71, 85, 105, 0.5);
            }
            
            .status-card-large {
                background: transparent;
                padding: 24px;
            }
            
            .status-card-title {
                color: rgb(34, 211, 238);
                font-weight: 600;
                font-size: 14px;
                margin-bottom: 10px;
                text-transform: uppercase;
                letter-spacing: 1px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .detection-title {
                font-size: 18px;
                color: rgb(34, 211, 238);
                font-weight: 700;
            }
            
            .detection-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 16px;
                margin-top: 16px;
            }
            
            .detection-item {
                background: rgba(15, 23, 42, 0.6);
                padding: 14px;
                border-radius: 6px;
                border: 1px solid rgba(71, 85, 105, 0.5);
            }
            
            .detection-label {
                color: rgb(148, 163, 184);
                font-size: 11px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                margin-bottom: 6px;
            }
            
            .detection-value {
                color: rgb(226, 232, 240);
                font-size: 20px;
                font-weight: 700;
                font-family: 'JetBrains Mono', monospace;
            }
            
            .detection-sub {
                color: rgb(100, 116, 139);
                font-size: 11px;
                margin-top: 4px;
                font-family: 'JetBrains Mono', monospace;
            }
            
            .collapsible-btn {
                width: 100%;
                background: rgba(15, 23, 42, 0.6);
                border: 1px solid rgba(71, 85, 105, 0.5);
                color: rgb(226, 232, 240);
                padding: 14px 16px;
                border-radius: 6px;
                cursor: pointer;
                font-size: 13px;
                font-weight: 600;
                text-align: left;
                display: flex;
                align-items: center;
                justify-content: space-between;
                transition: all 0.2s ease;
                margin: 0;
                text-transform: none;
                letter-spacing: 0;
            }
            
            .collapsible-btn:hover {
                background: rgba(15, 23, 42, 0.8);
                border-color: rgba(71, 85, 105, 0.8);
                transform: none;
            }
            
            .collapsible-btn .icon {
                font-size: 16px;
            }
            
            .collapsible-btn .arrow {
                transition: transform 0.2s ease;
                font-size: 12px;
            }
            
            .collapsible-btn.active .arrow {
                transform: rotate(180deg);
            }
            
            .collapsible-content {
                max-height: 0;
                overflow: hidden;
                transition: max-height 0.3s ease;
                background: rgba(10, 15, 24, 0.6);
                border-radius: 0 0 6px 6px;
                margin-top: -1px;
            }
            
            .collapsible-content.open {
                max-height: 3000px;
                border: 1px solid rgba(71, 85, 105, 0.5);
                border-top: none;
            }
            
            .collapsible-inner {
                padding: 16px;
            }
            
            .status-card p {
                margin: 6px 0;
                color: rgb(148, 163, 184);
                line-height: 1.6;
                font-size: 13px;
                font-family: 'JetBrains Mono', monospace;
            }
            
            .status-card strong {
                color: rgb(226, 232, 240);
                font-weight: 600;
            }
            
            .param-guide {
                background: linear-gradient(135deg, rgba(6, 182, 212, 0.05) 0%, rgba(6, 182, 212, 0.02) 100%);
                padding: 20px;
                border-radius: 8px;
                margin-bottom: 20px;
                border: 1px solid rgba(6, 182, 212, 0.2);
            }
            
            .param-guide-title {
                color: rgb(34, 211, 238);
                font-weight: 700;
                font-size: 16px;
                margin-bottom: 16px;
                display: flex;
                align-items: center;
                gap: 8px;
            }
            
            .param-item {
                margin: 12px 0;
                padding: 12px;
                background: rgba(15, 23, 42, 0.4);
                border-left: 3px solid;
                border-radius: 4px;
            }
            
            .param-item.safe {
                border-color: rgb(52, 211, 153);
            }
            
            .param-item.unsafe {
                border-color: rgb(248, 113, 113);
            }
            
            .param-item.threshold {
                border-color: rgb(251, 191, 36);
            }
            
            .param-item strong {
                display: block;
                font-size: 13px;
                margin-bottom: 4px;
                font-weight: 600;
            }
            
            .param-item.safe strong {
                color: rgb(52, 211, 153);
            }
            
            .param-item.unsafe strong {
                color: rgb(248, 113, 113);
            }
            
            .param-item.threshold strong {
                color: rgb(251, 191, 36);
            }
            
            .param-item span {
                color: rgb(148, 163, 184);
                font-size: 12px;
                line-height: 1.5;
                font-family: 'Inter', sans-serif;
            }
            
            .input-group {
                margin-bottom: 20px;
            }
            
            .input-group:last-child {
                margin-bottom: 0;
            }
            
            .input-row {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 16px;
                margin-bottom: 16px;
            }
            
            @media (max-width: 768px) {
                .input-row {
                    grid-template-columns: 1fr;
                }
                
                .input-group > div[style*="grid-template-columns: 1fr 1fr"] {
                    grid-template-columns: 1fr !important;
                }
                
                /* Stack status cards on mobile */
                .status-grid > div[style*="grid-template-columns: 1fr 2fr 1fr"] {
                    grid-template-columns: 1fr !important;
                }
            }
            
            @media (max-width: 1200px) {
                .section > div[style*="grid-template-columns"] {
                    grid-template-columns: 1fr !important;
                }
                
                .param-guide {
                    position: static !important;
                    margin-bottom: 24px;
                }
            }
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
            
            /* Safety Configuration Boxes */
            .safety-box {
                background: rgba(10, 15, 24, 0.9);
                border-radius: 8px;
                border: 1px solid rgba(71, 85, 105, 0.5);
                overflow: hidden;
            }
            
            .safety-box-header {
                padding: 16px;
                font-weight: 600;
                font-size: 14px;
                text-transform: uppercase;
                letter-spacing: 1px;
                display: flex;
                align-items: center;
                gap: 8px;
                border-bottom: 2px solid;
            }
            
            .safe-box {
                border-color: rgba(52, 211, 153, 0.3);
            }
            
            .safe-header {
                background: rgba(52, 211, 153, 0.1);
                color: rgb(52, 211, 153);
                border-bottom-color: rgba(52, 211, 153, 0.3);
            }
            
            .unsafe-box {
                border-color: rgba(248, 113, 113, 0.3);
            }
            
            .unsafe-header {
                background: rgba(248, 113, 113, 0.1);
                color: rgb(248, 113, 113);
                border-bottom-color: rgba(248, 113, 113, 0.3);
            }
            
            .safety-box-content {
                padding: 16px;
                max-height: 400px;
                overflow-y: auto;
            }
            
            .condition-item {
                padding: 12px;
                margin-bottom: 12px;
                background: rgba(15, 23, 42, 0.6);
                border-radius: 6px;
                border: 1px solid rgba(255, 255, 255, 0.08);
                transition: all 0.3s ease;
            }
            
            .condition-item:last-child {
                margin-bottom: 0;
            }
            
            .safe-condition:hover {
                background: rgba(52, 211, 153, 0.05);
                border-color: rgba(52, 211, 153, 0.3);
            }
            
            .unsafe-condition:hover {
                background: rgba(248, 113, 113, 0.05);
                border-color: rgba(248, 113, 113, 0.3);
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
            
            <div class="main-content">
                {% if message %}
                <div class="message">✓ {{ message }}</div>
                {% endif %}
                
                <!-- Status Overview Section -->
                <div class="section">
                    <div class="section-header">
                        <span class="icon">📊</span>
                        <span>System <span class="highlight">Status</span></span>
                    </div>
                    
                    <div class="status-grid">
                        <!-- Current Detection - Prominent -->
                        <div style="display: grid; grid-template-columns: 1fr 2fr 1fr; gap: 16px;">
                            <!-- Latest Detection Image -->
                            <div class="status-card">
                                <div class="status-card-title">📷 Latest Image</div>
                                <div style="display: flex; justify-content: center; align-items: center; padding: 8px; min-height: 250px;">
                                    <img id="latest-image" src="/api/v1/latest_image?t={{ last_update }}" 
                                         alt="Latest Detection" 
                                         style="width: 100%; height: 100%; object-fit: contain; border-radius: 6px; border: 1px solid rgba(71, 85, 105, 0.5);"
                                         onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';">
                                    <div style="display: none; justify-content: center; align-items: center; text-align: center; color: rgb(148, 163, 184); font-size: 12px; padding: 20px; width: 100%; height: 100%;">
                                        No image available
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Current Detection - Prominent -->
                            <div class="status-card status-card-large">
                                <div class="detection-grid" style="margin-top: 0; height: 100%;">
                                    <div class="detection-item">
                                        <div class="detection-label">Condition</div>
                                        <div class="detection-value">{{ current_condition }}</div>
                                    </div>
                                    <div class="detection-item">
                                        <div class="detection-label">Confidence</div>
                                        <div class="detection-value">{{ current_confidence }}%</div>
                                    </div>
                                    <div class="detection-item">
                                        <div class="detection-label">ASCOM Status</div>
                                        <div class="detection-value" style="color: {{ ascom_safe_color }};">{{ ascom_safe_status }}</div>
                                    </div>
                                    <div class="detection-item">
                                        <div class="detection-label">Detection Time</div>
                                        <div class="detection-value">{{ detection_time }}s</div>
                                    </div>
                                    <div class="detection-item">
                                        <div class="detection-label">Last Updated</div>
                                        <div class="detection-value" style="font-size: 14px;">{{ last_update }}</div>
                                    </div>
                                    <div class="detection-item">
                                        <div class="detection-label">Container Uptime</div>
                                        <div class="detection-value" style="font-size: 14px;">{{ container_uptime }}</div>
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Safety State History -->
                            <div class="status-card">
                                <div class="status-card-title">📜 Safety History</div>
                                <div style="max-height: 250px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: 11px;">
                                    {% if safety_history %}
                                        {% for entry in safety_history %}
                                        <div style="padding: 4px 8px; border-bottom: 1px solid rgba(71, 85, 105, 0.3); display: flex; justify-content: space-between; align-items: center;">
                                            <span style="color: {{ 'rgb(52, 211, 153)' if entry.is_safe else 'rgb(248, 113, 113)' }}; font-weight: 600;">
                                                {{ '✓ SAFE' if entry.is_safe else '⚠ UNSAFE' }}
                                            </span>
                                            <span style="color: rgb(148, 163, 184); font-size: 10px;">{{ entry.time }}</span>
                                        </div>
                                        <div style="padding: 2px 8px 6px 8px; font-size: 10px; color: rgb(148, 163, 184);">
                                            {{ entry.condition }} ({{ entry.confidence }}%)
                                        </div>
                                        {% endfor %}
                                    {% else %}
                                        <div style="padding: 12px; text-align: center; color: rgb(148, 163, 184); font-size: 12px;">
                                            No state changes yet
                                        </div>
                                    {% endif %}
                                </div>
                            </div>
                        </div>
                        
                        <!-- ASCOM Connection & Device Info - Horizontal Layout -->
                        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px;">
                            <!-- Client Connection - Collapsible -->
                            <div>
                                <button class="collapsible-btn" onclick="toggleCollapsible('ascom-details')">
                                    <span><span class="icon">🔌</span> ASCOM Connection: <span class="{{ ascom_status_class }}">{{ ascom_status }}</span></span>
                                    <span class="arrow">▼</span>
                                </button>
                                <div id="ascom-details" class="collapsible-content">
                                    <div class="collapsible-inner">
                                        <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                            Status: <span class="{{ ascom_status_class }}">{{ ascom_status }}</span>
                                        </p>
                                        {% if client_ip %}
                                        <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                            Client: <strong style="color: rgb(226, 232, 240);">{{ client_ip }}</strong>
                                        </p>
                                        {% endif %}
                                        {% if ascom_status == 'Connected' %}
                                            {% if connection_duration %}
                                            <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                                Duration: <strong style="color: rgb(226, 232, 240);">{{ connection_duration }}</strong>
                                            </p>
                                            {% endif %}
                                        {% else %}
                                            {% if last_session_duration %}
                                            <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                                Last Session Duration: <strong style="color: rgb(226, 232, 240);">{{ last_session_duration }}</strong>
                                            </p>
                                            {% endif %}
                                        {% endif %}
                                        {% if last_connected %}
                                        <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                            Last Connected: <strong style="color: rgb(226, 232, 240);">{{ last_connected }}</strong>
                                        </p>
                                        {% endif %}
                                        {% if last_disconnected %}
                                        <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                            Last Disconnected: <strong style="color: rgb(226, 232, 240);">{{ last_disconnected }}</strong>
                                        </p>
                                        {% endif %}
                                    </div>
                                </div>
                            </div>
                            
                            <!-- Device Info - Collapsible -->
                            <div>
                                <button class="collapsible-btn" onclick="toggleCollapsible('device-details')">
                                    <span><span class="icon">⚙️</span> Device Information</span>
                                    <span class="arrow">▼</span>
                                </button>
                                <div id="device-details" class="collapsible-content">
                                    <div class="collapsible-inner">
                                        <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                            Name: <strong style="color: rgb(226, 232, 240);">{{ current_name }}</strong>
                                        </p>
                                        <p style="margin: 6px 0; color: rgb(148, 163, 184); font-size: 13px; font-family: 'JetBrains Mono', monospace;">
                                            Location: <strong style="color: rgb(226, 232, 240);">{{ current_location }}</strong>
                                        </p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
                
                <script>
                function toggleCollapsible(id) {
                    const content = document.getElementById(id);
                    const btn = content.previousElementSibling;
                    
                    if (content.classList.contains('open')) {
                        content.classList.remove('open');
                        btn.classList.remove('active');
                    } else {
                        content.classList.add('open');
                        btn.classList.add('active');
                    }
                }
                
                // Auto-refresh system status
                function updateSystemStatus() {
                    fetch(window.location.href)
                        .then(response => response.text())
                        .then(html => {
                            const parser = new DOMParser();
                            const doc = parser.parseFromString(html, 'text/html');
                            
                            // Update Current Detection - all fields including Detection Time and Last Updated
                            const detectionGrid = document.querySelector('.detection-grid');
                            const newDetectionGrid = doc.querySelector('.detection-grid');
                            if (detectionGrid && newDetectionGrid) {
                                detectionGrid.innerHTML = newDetectionGrid.innerHTML;
                            }
                            
                            // Update Safety History - find by looking for the title text
                            const statusCards = document.querySelectorAll('.status-card');
                            const newStatusCards = doc.querySelectorAll('.status-card');
                            
                            statusCards.forEach((card, index) => {
                                const title = card.querySelector('.status-card-title');
                                if (title && title.textContent.includes('📜 Safety History')) {
                                    const scrollableDiv = card.querySelector('div[style*="overflow-y"]');
                                    const oldScrollTop = scrollableDiv ? scrollableDiv.scrollTop : 0;
                                    
                                    const newCard = newStatusCards[index];
                                    if (newCard) {
                                        card.innerHTML = newCard.innerHTML;
                                        const newScrollableDiv = card.querySelector('div[style*="overflow-y"]');
                                        if (newScrollableDiv) {
                                            newScrollableDiv.scrollTop = oldScrollTop;
                                        }
                                    }
                                }
                            });
                            
                            // Update ASCOM Connection details including Duration
                            const ascomDetails = document.getElementById('ascom-details');
                            const newAscomDetails = doc.getElementById('ascom-details');
                            if (ascomDetails && newAscomDetails) {
                                const wasOpen = ascomDetails.classList.contains('open');
                                ascomDetails.innerHTML = newAscomDetails.innerHTML;
                                if (wasOpen) {
                                    ascomDetails.classList.add('open');
                                }
                            }
                            
                            // Update ASCOM Connection button status text
                            const ascomButton = document.querySelector('button[onclick*="ascom-details"]');
                            const newAscomButton = doc.querySelector('button[onclick*="ascom-details"]');
                            if (ascomButton && newAscomButton) {
                                const statusSpan = ascomButton.querySelector('span[class*="status-"]');
                                const newStatusSpan = newAscomButton.querySelector('span[class*="status-"]');
                                if (statusSpan && newStatusSpan) {
                                    statusSpan.className = newStatusSpan.className;
                                    statusSpan.textContent = newStatusSpan.textContent;
                                }
                            }
                            
                            // Update Latest Image - refresh with timestamp to avoid cache
                            const latestImage = document.getElementById('latest-image');
                            if (latestImage) {
                                const timestamp = new Date().getTime();
                                latestImage.src = '/api/v1/latest_image?t=' + timestamp;
                            }
                        })
                        .catch(error => console.error('Error updating status:', error));
                }
                
                // Update every 5 seconds
                setInterval(updateSystemStatus, 5000);
                </script>
                
                <!-- Configuration Form Section -->
                <div class="section">
                    <div class="section-header">
                        <span class="icon">🛠️</span>
                        <span>Configuration <span class="highlight">Settings</span></span>
                    </div>
                    
                    <form method="POST">
                                <!-- Parameter Guide (moved from sidebar) -->
                                <div style="margin-bottom: 20px;">
                                    <button type="button" class="collapsible-btn" onclick="toggleCollapsible('param-guide')">
                                        <span><span class="icon">📖</span> Parameter Guide</span>
                                        <span class="arrow">▼</span>
                                    </button>
                                    <div id="param-guide" class="collapsible-content">
                                        <div class="collapsible-inner">
                                            <div class="param-item" style="border-color: rgb(59, 130, 246);">
                                                <strong style="color: rgb(59, 130, 246);">Image Fetch Interval</strong>
                                                <span>How often to download new images from AllSky camera and run AI analysis. Lower values = faster response but higher CPU usage.</span>
                                            </div>
                                            <div style="margin: 8px 0 12px 12px; padding: 8px; background: rgba(59, 130, 246, 0.05); border-radius: 4px;">
                                                <div style="font-size: 11px; color: rgb(148, 163, 184); line-height: 1.5;">
                                                    <strong style="color: rgb(59, 130, 246); font-size: 12px;">Recommended:</strong><br>
                                                    • Fast response: 15-30s<br>
                                                    • Balanced: 30-60s<br>
                                                    • Resource efficient: 60-120s
                                                </div>
                                            </div>
                                            
                                            <div class="param-item" style="border-color: rgb(236, 72, 153);">
                                                <strong style="color: rgb(236, 72, 153);">ASCOM Update Interval</strong>
                                                <span>How often to re-check safety status. Should be ≥ Image Fetch Interval.</span>
                                            </div>
                                            <div style="margin: 8px 0 12px 12px; padding: 8px; background: rgba(236, 72, 153, 0.05); border-radius: 4px;">
                                                <div style="font-size: 11px; color: rgb(148, 163, 184); line-height: 1.5;">
                                                    <strong style="color: rgb(236, 72, 153); font-size: 12px;">Rule:</strong> Set equal to Image Fetch Interval for efficiency.
                                                </div>
                                            </div>
                                            
                                            <div class="param-item safe">
                                                <strong>Safe Wait Time</strong>
                                                <span>How long the sky must remain clear before the system reports "Safe".</span>
                                            </div>
                                            <div style="margin: 8px 0 12px 12px; padding: 8px; background: rgba(52, 211, 153, 0.05); border-radius: 4px;">
                                                <div style="font-size: 11px; color: rgb(148, 163, 184); line-height: 1.5;">
                                                    <strong style="color: rgb(52, 211, 153); font-size: 12px;">Why it matters:</strong><br>
                                                    Prevents opening the observatory roof during brief breaks in storms or passing clouds. 
                                                    Set higher (e.g., 300s = 5 min) for unstable weather patterns.<br><br>
                                                    <strong style="color: rgb(52, 211, 153); font-size: 12px;">Recommended:</strong><br>
                                                    • Stable climate: 60-120s<br>
                                                    • Variable weather: 180-300s<br>
                                                    • Storm-prone areas: 300-600s
                                                </div>
                                            </div>
                                            
                                            <div class="param-item unsafe">
                                                <strong>Unsafe Wait Time (Debounce to Unsafe)</strong>
                                                <span>How long bad weather must persist before the system reports "Unsafe".</span>
                                            </div>
                                            <div style="margin: 8px 0 12px 12px; padding: 8px; background: rgba(248, 113, 113, 0.05); border-radius: 4px;">
                                                <div style="font-size: 11px; color: rgb(148, 163, 184); line-height: 1.5;">
                                                    <strong style="color: rgb(248, 113, 113); font-size: 12px;">Why it matters:</strong><br>
                                                    Controls emergency response speed. Setting to 0 triggers immediate roof closure at first sign of danger. 
                                                    Higher values (10-30s) can filter out brief sensor glitches.<br><br>
                                                    <strong style="color: rgb(248, 113, 113); font-size: 12px;">Recommended:</strong><br>
                                                    • Automated roof: 0s (immediate)<br>
                                                    • Manual operation: 5-10s<br>
                                                    • Glitch filtering: 10-30s
                                                </div>
                                            </div>
                                            
                                            <div class="param-item threshold">
                                                <strong>Confidence Threshold</strong>
                                                <span>Minimum AI confidence (%) required to trigger each weather condition.</span>
                                            </div>
                                            <div style="margin: 8px 0 12px 12px; padding: 8px; background: rgba(251, 191, 36, 0.05); border-radius: 4px;">
                                                <div style="font-size: 11px; color: rgb(148, 163, 184); line-height: 1.5;">
                                                    <strong style="color: rgb(251, 191, 36); font-size: 12px;">📊 Key Concept:</strong><br>
                                                    <strong>Higher threshold = Less sensitive</strong> (AI must be more confident)<br>
                                                    <strong>Lower threshold = More sensitive</strong> (AI triggers with less certainty)<br><br>
                                                    
                                                    <strong style="color: rgb(251, 191, 36); font-size: 12px;">How it works:</strong><br>
                                                    If set to 80% for "Rain", the AI must be 80% confident it's raining to trigger. 
                                                    Lower = more sensitive (fewer misses, more false alarms). Higher = less sensitive (more misses, fewer false alarms).<br><br>
                                                    <strong style="color: rgb(251, 191, 36); font-size: 12px;">Tuning tips:</strong><br>
                                                    • Start at 50% (default)<br>
                                                    • Lower for critical conditions (Rain: 40-50%)<br>
                                                    • Higher for uncertain conditions (Overcast: 60-70%)<br>
                                                    • Adjust based on false alarm rate
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Basic Settings -->
                                <div style="margin-bottom: 20px;">
                                    <button type="button" class="collapsible-btn" onclick="toggleCollapsible('basic-settings')">
                                        <span><span class="icon">⚙️</span> Basic Settings</span>
                                        <span class="arrow">▼</span>
                                    </button>
                                    <div id="basic-settings" class="collapsible-content">
                                        <div class="collapsible-inner">
                                            <div class="input-group" style="margin: 0;">
                                                <div class="input-row">
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
                                                </div>
                                                <div class="input-row">
                                                    <div class="form-group" style="grid-column: span 2;">
                                                        <label for="image_url">Image Source URL</label>
                                                        <input type="text" id="image_url" name="image_url" class="glow-input"
                                                               value="{{ current_image_url }}" placeholder="{{ image_url_default }}">
                                                        <small style="color: rgb(148, 163, 184); font-size: 12px; margin-top: 4px; display: block;">
                                                            URL to fetch images for cloud detection. Leave empty to use environment variable default.
                                                        </small>
                                                    </div>
                                                </div>
                                                
                                                <div class="input-row">
                                                    <div class="form-group">
                                                        <label for="ntp_server">NTP Server</label>
                                                        <input type="text" id="ntp_server" name="ntp_server" class="glow-input"
                                                               value="{{ current_ntp_server }}" placeholder="pool.ntp.org">
                                                        <small style="color: rgb(148, 163, 184); font-size: 12px; margin-top: 4px; display: block;">
                                                            Time server for accurate timestamps
                                                        </small>
                                                    </div>
                                                    
                                                    <div class="form-group">
                                                        <label for="timezone">Timezone</label>
                                                        <input type="text" id="timezone" name="timezone" class="glow-input"
                                                               value="{{ current_timezone }}" placeholder="UTC">
                                                        <small style="color: rgb(148, 163, 184); font-size: 12px; margin-top: 4px; display: block;">
                                                            e.g., America/New_York, Europe/London, UTC
                                                            <a href="https://en.wikipedia.org/wiki/List_of_tz_database_time_zones" target="_blank" style="color: rgb(6, 182, 212); text-decoration: none; margin-left: 8px;">📖 Reference</a>
                                                        </small>
                                                    </div>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Timing Configuration -->
                                <div style="margin-bottom: 20px;">
                                    <button type="button" class="collapsible-btn" onclick="toggleCollapsible('timing-config')">
                                        <span><span class="icon">⏱️</span> Timing Configuration</span>
                                        <span class="arrow">▼</span>
                                    </button>
                                    <div id="timing-config" class="collapsible-content">
                                        <div class="collapsible-inner">
                                            <div class="input-group" style="margin: 0;">
                                    
                                    <!-- How It Works Visualization -->
                                    <div style="margin-bottom: 24px; padding: 20px; background: rgba(15, 23, 42, 0.8); border: 1px solid rgb(51, 65, 85); border-radius: 8px;">
                                        <h3 style="color: rgb(226, 232, 240); font-size: 14px; font-weight: 600; margin-bottom: 16px;">How Timing Works</h3>
                                        
                                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 20px;">
                                            <!-- Fetch Image Card -->
                                            <div style="background: rgba(59, 130, 246, 0.1); border: 2px solid rgba(59, 130, 246, 0.4); border-radius: 8px; padding: 16px;">
                                                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                                                    <div style="width: 40px; height: 40px; background: rgb(59, 130, 246); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">📷</div>
                                                    <div>
                                                        <div style="color: rgb(59, 130, 246); font-weight: 600; font-size: 13px;">FETCH IMAGE</div>
                                                        <div id="fetch-display" style="color: rgb(148, 163, 184); font-size: 11px;">Every 60s</div>
                                                    </div>
                                                </div>
                                                <div style="color: rgb(203, 213, 225); font-size: 12px; line-height: 1.5;">
                                                    Downloads latest image from AllSky camera and runs AI analysis to detect weather conditions
                                                </div>
                                            </div>
                                            
                                            <!-- ASCOM Update Card -->
                                            <div style="background: rgba(236, 72, 153, 0.1); border: 2px solid rgba(236, 72, 153, 0.4); border-radius: 8px; padding: 16px;">
                                                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                                                    <div style="width: 40px; height: 40px; background: rgb(236, 72, 153); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">🔄</div>
                                                    <div>
                                                        <div style="color: rgb(236, 72, 153); font-weight: 600; font-size: 13px;">ASCOM UPDATE</div>
                                                        <div id="ascom-display" style="color: rgb(148, 163, 184); font-size: 11px;">Every 30s</div>
                                                    </div>
                                                </div>
                                                <div style="color: rgb(203, 213, 225); font-size: 12px; line-height: 1.5;">
                                                    Re-checks safety status using latest detection result and applies debounce logic
                                                </div>
                                            </div>
                                            
                                            <!-- Safe Debounce Card -->
                                            <div style="background: rgba(52, 211, 153, 0.1); border: 2px solid rgba(52, 211, 153, 0.4); border-radius: 8px; padding: 16px;">
                                                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                                                    <div style="width: 40px; height: 40px; background: rgb(52, 211, 153); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">✓</div>
                                                    <div>
                                                        <div style="color: rgb(52, 211, 153); font-weight: 600; font-size: 13px;">SAFE WAIT TIME</div>
                                                        <div id="safe-display" style="color: rgb(148, 163, 184); font-size: 11px;">60s</div>
                                                    </div>
                                                </div>
                                                <div style="color: rgb(203, 213, 225); font-size: 12px; line-height: 1.5;">
                                                    How long sky must <strong>stay clear</strong> before reporting "Safe" - prevents premature roof opening
                                                </div>
                                                <div id="safe-coverage" style="margin-top: 8px; padding: 8px; background: rgba(52, 211, 153, 0.1); border-radius: 4px; color: rgb(148, 163, 184); font-size: 11px;"></div>
                                            </div>
                                            
                                            <!-- Unsafe Debounce Card -->
                                            <div style="background: rgba(248, 113, 113, 0.1); border: 2px solid rgba(248, 113, 113, 0.4); border-radius: 8px; padding: 16px;">
                                                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
                                                    <div style="width: 40px; height: 40px; background: rgb(248, 113, 113); border-radius: 8px; display: flex; align-items: center; justify-content: center; font-size: 20px;">⚡</div>
                                                    <div>
                                                        <div style="color: rgb(248, 113, 113); font-weight: 600; font-size: 13px;">UNSAFE WAIT TIME</div>
                                                        <div id="unsafe-display" style="color: rgb(148, 163, 184); font-size: 11px;">Immediate</div>
                                                    </div>
                                                </div>
                                                <div style="color: rgb(203, 213, 225); font-size: 12px; line-height: 1.5;">
                                                    How long bad weather must persist before reporting "Unsafe" - 0 = instant roof closure
                                                </div>
                                                <div id="unsafe-coverage" style="margin-top: 8px; padding: 8px; background: rgba(248, 113, 113, 0.1); border-radius: 4px; color: rgb(148, 163, 184); font-size: 11px;"></div>
                                            </div>
                                        </div>
                                        
                                        <!-- Example Scenario -->
                                        <div id="scenario-example" style="margin-top: 20px; padding: 16px; background: rgba(30, 41, 59, 0.6); border-radius: 8px; border-left: 4px solid rgb(251, 191, 36);">
                                            <div style="color: rgb(251, 191, 36); font-weight: 600; font-size: 12px; margin-bottom: 8px;">💡 EXAMPLE SCENARIO</div>
                                            <div style="color: rgb(203, 213, 225); font-size: 12px; line-height: 1.6;"></div>
                                        </div>
                                    </div>
                                    
                                    <div class="input-row">
                                        <div class="form-group">
                                            <label for="detection_interval">Image Fetch Interval (seconds)</label>
                                            <input type="number" id="detection_interval" name="detection_interval" 
                                                   value="{{ detection_interval }}" min="5" max="300" step="1" placeholder="30"
                                                   style="width: 100%; padding: 12px 16px; background: rgba(15, 23, 42, 0.6); 
                                                          border: 1px solid rgb(71, 85, 105); border-radius: 6px; 
                                                          color: rgb(241, 245, 249); font-size: 14px; font-family: 'JetBrains Mono', monospace;
                                                          transition: all 0.2s ease;">
                                            <p class="help-text">How often to fetch & analyze new images from AllSky camera</p>
                                        </div>
                                        
                                        <div class="form-group">
                                            <label for="update_interval">ASCOM Update Interval (seconds)</label>
                                            <input type="number" id="update_interval" name="update_interval" 
                                                   value="{{ update_interval }}" min="5" max="300" step="1" placeholder="30"
                                                   style="width: 100%; padding: 12px 16px; background: rgba(15, 23, 42, 0.6); 
                                                          border: 1px solid rgb(71, 85, 105); border-radius: 6px; 
                                                          color: rgb(241, 245, 249); font-size: 14px; font-family: 'JetBrains Mono', monospace;
                                                          transition: all 0.2s ease;">
                                            <p class="help-text">How often to check safety status</p>
                                        </div>
                                    </div>
                                    
                                    <div class="input-row">
                                        <div class="form-group">
                                            <label for="debounce_safe" style="color: rgb(52, 211, 153);">Safe Wait Time (seconds)</label>
                                            <input type="number" id="debounce_safe" name="debounce_safe" 
                                                   value="{{ debounce_to_safe }}" min="0" max="3600" step="1" placeholder="60"
                                                   style="width: 100%; padding: 12px 16px; background: rgba(15, 23, 42, 0.6); 
                                                          border: 1px solid rgb(71, 85, 105); border-radius: 6px; 
                                                          color: rgb(241, 245, 249); font-size: 14px; font-family: 'JetBrains Mono', monospace;
                                                          transition: all 0.2s ease;">
                                            <p class="help-text">Delay before reporting safe conditions</p>
                                        </div>
                                        
                                        <div class="form-group">
                                            <label for="debounce_unsafe" style="color: rgb(248, 113, 113);">Unsafe Wait Time (seconds)</label>
                                            <input type="number" id="debounce_unsafe" name="debounce_unsafe" 
                                                   value="{{ debounce_to_unsafe }}" min="0" max="3600" step="1" placeholder="0"
                                                   style="width: 100%; padding: 12px 16px; background: rgba(15, 23, 42, 0.6); 
                                                          border: 1px solid rgb(71, 85, 105); border-radius: 6px; 
                                                          color: rgb(241, 245, 249); font-size: 14px; font-family: 'JetBrains Mono', monospace;
                                                          transition: all 0.2s ease;">
                                            <p class="help-text">Delay before reporting unsafe conditions (0 = immediate)</p>
                                        </div>
                                    </div>
                                    
                                    <script>
                                    function updateTimingVisualization() {
                                        const detectionInterval = parseInt(document.getElementById('detection_interval').value) || 30;
                                        const updateInterval = parseInt(document.getElementById('update_interval').value) || 30;
                                        const debounceSafe = parseInt(document.getElementById('debounce_safe').value) || 60;
                                        const debounceUnsafe = parseInt(document.getElementById('debounce_unsafe').value) || 0;
                                        
                                        // Update displays
                                        document.getElementById('fetch-display').textContent = `Every ${detectionInterval}s`;
                                        document.getElementById('ascom-display').textContent = `Every ${updateInterval}s`;
                                        document.getElementById('safe-display').textContent = debounceSafe > 0 ? `${debounceSafe}s` : 'None';
                                        document.getElementById('unsafe-display').textContent = debounceUnsafe > 0 ? `${debounceUnsafe}s` : 'Immediate';
                                        
                                        // Calculate coverage info
                                        if (debounceSafe > 0) {
                                            const checksNeeded = Math.ceil(debounceSafe / detectionInterval);
                                            document.getElementById('safe-coverage').innerHTML = 
                                                `<strong>Requires ${checksNeeded} consecutive clear detections</strong> (${checksNeeded} × ${detectionInterval}s = ${checksNeeded * detectionInterval}s)`;
                                        } else {
                                            document.getElementById('safe-coverage').textContent = 'Reports safe immediately after first clear detection';
                                        }
                                        
                                        if (debounceUnsafe > 0) {
                                            const checksNeeded = Math.ceil(debounceUnsafe / detectionInterval);
                                            document.getElementById('unsafe-coverage').innerHTML = 
                                                `<strong>Requires ${checksNeeded} consecutive bad detections</strong> (${checksNeeded} × ${detectionInterval}s = ${checksNeeded * detectionInterval}s)`;
                                        } else {
                                            document.getElementById('unsafe-coverage').textContent = 'Reports unsafe immediately - closes roof on first bad detection';
                                        }
                                        
                                        // Generate example scenario
                                        const scenarioDiv = document.querySelector('#scenario-example > div:last-child');
                                        let scenario = '';
                                        
                                        if (updateInterval < detectionInterval) {
                                            scenario = `⚠️ <strong>Warning:</strong> ASCOM checks every ${updateInterval}s but new data only arrives every ${detectionInterval}s. You'll check the same result ${Math.floor(detectionInterval/updateInterval)} times before getting new data.`;
                                        } else {
                                            // Safe scenario
                                            let safeScenario = '';
                                            if (debounceSafe > 0) {
                                                const checksNeeded = Math.ceil(debounceSafe / detectionInterval);
                                                safeScenario = `Clouds clear at 00:00 → AI detects "Clear" every ${detectionInterval}s → After ${checksNeeded} detections (${checksNeeded * detectionInterval}s total) → Reports <strong style="color: rgb(52, 211, 153);">SAFE</strong> at ${formatTime(checksNeeded * detectionInterval)}`;
                                            } else {
                                                safeScenario = `Clouds clear at 00:00 → AI detects "Clear" at ${formatTime(detectionInterval)} → Immediately reports <strong style="color: rgb(52, 211, 153);">SAFE</strong>`;
                                            }
                                            
                                            // Unsafe scenario
                                            let unsafeScenario = '';
                                            if (debounceUnsafe > 0) {
                                                const checksNeeded = Math.ceil(debounceUnsafe / detectionInterval);
                                                unsafeScenario = `<br><br>Rain detected at 00:00 → AI detects "Rain" every ${detectionInterval}s → After ${checksNeeded} detections (${checksNeeded * detectionInterval}s total) → Reports <strong style="color: rgb(248, 113, 113);">UNSAFE</strong> at ${formatTime(checksNeeded * detectionInterval)}`;
                                            } else {
                                                unsafeScenario = `<br><br>Rain detected at 00:00 → AI detects "Rain" at ${formatTime(detectionInterval)} → ⚡ Immediately reports <strong style="color: rgb(248, 113, 113);">UNSAFE</strong> (roof closes!)`;
                                            }
                                            
                                            scenario = safeScenario + unsafeScenario;
                                        }
                                        
                                        scenarioDiv.innerHTML = scenario;
                                    }
                                    
                                    function formatTime(seconds) {
                                        const mins = Math.floor(seconds / 60);
                                        const secs = seconds % 60;
                                        return mins > 0 ? `${String(mins).padStart(2, '0')}:${String(secs).padStart(2, '0')}` : `00:${String(secs).padStart(2, '0')}`;
                                    }
                                    </script>
                                    
                                    <script>
                                    // Validate timing configuration for safety
                                    function validateTiming() {
                                        // Update timing visualization
                                        updateTimingVisualization();
                                    }
                                    
                                    document.addEventListener('DOMContentLoaded', function() {
                                        const timingInputs = ['detection_interval', 'update_interval', 'debounce_safe', 'debounce_unsafe'];
                                        timingInputs.forEach(id => {
                                            const input = document.getElementById(id);
                                            if (input) {
                                                input.addEventListener('input', validateTiming);
                                                input.addEventListener('change', validateTiming);
                                            }
                                        });
                                        validateTiming();
                                    });
                                    </script>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <!-- Safety Configuration -->
                                <div style="margin-bottom: 20px;">
                                    <button type="button" class="collapsible-btn" onclick="toggleCollapsible('safety-config')">
                                        <span><span class="icon">🛡️</span> Safety Classification</span>
                                        <span class="arrow">▼</span>
                                    </button>
                                    <div id="safety-config" class="collapsible-content">
                                        <div class="collapsible-inner">
                                            <div class="input-group" style="margin: 0;">
                                    <p class="help-text" style="margin-bottom: 16px;">
                                        Mark each weather condition as <strong style="color: rgb(52, 211, 153);">Safe</strong> or <strong style="color: rgb(248, 113, 113);">Unsafe</strong> for observatory operations.
                                    </p>
                                    
                                    <!-- Safe Conditions -->
                                    <div id="safe-section" style="margin-bottom: 20px;">
                                        <h3 style="color: rgb(52, 211, 153); font-size: 14px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Safe Conditions</h3>
                                        <div id="safe-conditions" style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(52, 211, 153, 0.3); border-radius: 8px; padding: 12px;">
                                            {% for condition in safe_conditions %}
                                            <div class="condition-card" data-condition="{{ condition }}" data-safety="safe" style="padding: 12px; background: rgba(30, 41, 59, 0.4); border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(71, 85, 105, 0.5);">
                                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                                                    <div style="font-weight: 600; color: rgb(226, 232, 240); font-size: 14px;">{{ condition }}</div>
                                                    <div style="display: flex; gap: 16px;">
                                                        <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                                            <input type="radio" name="safety_{{ condition }}" value="safe" checked 
                                                                   class="safety-radio" data-condition="{{ condition }}"
                                                                   style="margin: 0; accent-color: rgb(52, 211, 153); width: 16px; height: 16px;">
                                                            <span style="color: rgb(52, 211, 153); font-weight: 500; font-size: 13px;">Safe</span>
                                                        </label>
                                                        <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                                            <input type="radio" name="safety_{{ condition }}" value="unsafe"
                                                                   class="safety-radio" data-condition="{{ condition }}"
                                                                   style="margin: 0; accent-color: rgb(248, 113, 113); width: 16px; height: 16px;">
                                                            <span style="color: rgb(248, 113, 113); font-weight: 500; font-size: 13px;">Unsafe</span>
                                                        </label>
                                                    </div>
                                                </div>
                                                <div style="display: flex; align-items: center; gap: 12px;">
                                                    <label for="threshold_{{ condition }}" style="margin: 0; font-size: 11px; color: rgb(148, 163, 184); min-width: 80px;">Threshold:</label>
                                                    <input type="number" id="threshold_{{ condition }}" name="threshold_{{ condition }}"
                                                           value="{{ class_thresholds.get(condition, default_threshold) }}" min="0" max="100" step="1" placeholder="{{ default_threshold }}"
                                                           class="threshold-input" data-condition="{{ condition }}"
                                                           style="width: 60px; padding: 5px 8px; background: rgba(10, 15, 24, 0.9); border: 1px solid rgb(71, 85, 105); border-radius: 4px; color: rgb(241, 245, 249); font-size: 12px; font-family: 'JetBrains Mono', monospace;">
                                                    <span style="font-size: 11px; color: rgb(148, 163, 184); min-width: 16px;">%</span>
                                                    
                                                    <!-- Threshold Bar Visualization -->
                                                    <div style="flex: 1; position: relative; height: 20px; background: rgba(30, 41, 59, 0.6); border-radius: 3px; overflow: hidden; border: 1px solid rgba(71, 85, 105, 0.5);">
                                                        <div id="bar_below_{{ condition }}" style="position: absolute; left: 0; top: 0; bottom: 0; width: {{ class_thresholds.get(condition, default_threshold) }}%; background: rgba(100, 116, 139, 0.4); transition: width 0.2s ease;"></div>
                                                        <div id="bar_above_{{ condition }}" style="position: absolute; top: 0; bottom: 0; right: 0; width: {{ 100 - class_thresholds.get(condition, default_threshold) }}%; background: rgba(52, 211, 153, 0.6); transition: width 0.2s ease, background 0.2s ease;"></div>
                                                        <div id="bar_marker_{{ condition }}" style="position: absolute; top: 0; bottom: 0; left: {{ class_thresholds.get(condition, default_threshold) }}%; width: 2px; background: rgb(251, 191, 36); z-index: 10; transition: left 0.2s ease;">
                                                            <div style="position: absolute; top: -18px; left: 50%; transform: translateX(-50%); color: rgb(251, 191, 36); font-size: 9px; font-weight: 600; white-space: nowrap; background: rgba(15, 23, 42, 0.9); padding: 1px 4px; border-radius: 2px;">{{ class_thresholds.get(condition, default_threshold) }}%</div>
                                                        </div>
                                                    </div>
                                                    <div style="min-width: 70px; font-size: 10px; color: rgb(148, 163, 184); text-align: right;">
                                                        <span id="bar_label_{{ condition }}">✓ Triggers >{{ class_thresholds.get(condition, default_threshold) }}%</span>
                                                    </div>
                                                </div>
                                            </div>
                                            {% endfor %}
                                        </div>
                                    </div>
                                    
                                    <!-- Unsafe Conditions -->
                                    <div id="unsafe-section">
                                        <h3 style="color: rgb(248, 113, 113); font-size: 14px; font-weight: 600; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">Unsafe Conditions</h3>
                                        <div id="unsafe-conditions" style="background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(248, 113, 113, 0.3); border-radius: 8px; padding: 12px;">
                                            {% for condition in unsafe_conditions %}
                                            <div class="condition-card" data-condition="{{ condition }}" data-safety="unsafe" style="padding: 12px; background: rgba(30, 41, 59, 0.4); border-radius: 6px; margin-bottom: 8px; border: 1px solid rgba(71, 85, 105, 0.5);">
                                                <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 10px;">
                                                    <div style="font-weight: 600; color: rgb(226, 232, 240); font-size: 14px;">{{ condition }}</div>
                                                    <div style="display: flex; gap: 16px;">
                                                        <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                                            <input type="radio" name="safety_{{ condition }}" value="safe"
                                                                   class="safety-radio" data-condition="{{ condition }}"
                                                                   style="margin: 0; accent-color: rgb(52, 211, 153); width: 16px; height: 16px;">
                                                            <span style="color: rgb(52, 211, 153); font-weight: 500; font-size: 13px;">Safe</span>
                                                        </label>
                                                        <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                                                            <input type="radio" name="safety_{{ condition }}" value="unsafe" checked
                                                                   class="safety-radio" data-condition="{{ condition }}"
                                                                   style="margin: 0; accent-color: rgb(248, 113, 113); width: 16px; height: 16px;">
                                                            <span style="color: rgb(248, 113, 113); font-weight: 500; font-size: 13px;">Unsafe</span>
                                                        </label>
                                                    </div>
                                                </div>
                                                <div style="display: flex; align-items: center; gap: 12px;">
                                                    <label for="threshold_{{ condition }}" style="margin: 0; font-size: 11px; color: rgb(148, 163, 184); min-width: 80px;">Threshold:</label>
                                                    <input type="number" id="threshold_{{ condition }}" name="threshold_{{ condition }}"
                                                           value="{{ class_thresholds.get(condition, default_threshold) }}" min="0" max="100" step="1" placeholder="{{ default_threshold }}"
                                                           class="threshold-input" data-condition="{{ condition }}"
                                                           style="width: 60px; padding: 5px 8px; background: rgba(10, 15, 24, 0.9); border: 1px solid rgb(71, 85, 105); border-radius: 4px; color: rgb(241, 245, 249); font-size: 12px; font-family: 'JetBrains Mono', monospace;">
                                                    <span style="font-size: 11px; color: rgb(148, 163, 184); min-width: 16px;">%</span>
                                                    
                                                    <!-- Threshold Bar Visualization -->
                                                    <div style="flex: 1; position: relative; height: 20px; background: rgba(30, 41, 59, 0.6); border-radius: 3px; overflow: hidden; border: 1px solid rgba(71, 85, 105, 0.5);">
                                                        <div id="bar_below_{{ condition }}" style="position: absolute; left: 0; top: 0; bottom: 0; width: {{ class_thresholds.get(condition, default_threshold) }}%; background: rgba(100, 116, 139, 0.4); transition: width 0.2s ease;"></div>
                                                        <div id="bar_above_{{ condition }}" style="position: absolute; top: 0; bottom: 0; right: 0; width: {{ 100 - class_thresholds.get(condition, default_threshold) }}%; background: rgba(248, 113, 113, 0.6); transition: width 0.2s ease, background 0.2s ease;"></div>
                                                        <div id="bar_marker_{{ condition }}" style="position: absolute; top: 0; bottom: 0; left: {{ class_thresholds.get(condition, default_threshold) }}%; width: 2px; background: rgb(251, 191, 36); z-index: 10; transition: left 0.2s ease;">
                                                            <div style="position: absolute; top: -18px; left: 50%; transform: translateX(-50%); color: rgb(251, 191, 36); font-size: 9px; font-weight: 600; white-space: nowrap; background: rgba(15, 23, 42, 0.9); padding: 1px 4px; border-radius: 2px;">{{ class_thresholds.get(condition, default_threshold) }}%</div>
                                                        </div>
                                                    </div>
                                                    <div style="min-width: 70px; font-size: 10px; color: rgb(148, 163, 184); text-align: right;">
                                                        <span id="bar_label_{{ condition }}">✓ Triggers >{{ class_thresholds.get(condition, default_threshold) }}%</span>
                                                    </div>
                                                </div>
                                            </div>
                                            {% endfor %}
                                        </div>
                                    </div>
                                    
                                    <script>
                                    // Dynamic sorting when radio buttons change
                                    document.addEventListener('DOMContentLoaded', function() {
                                        const radios = document.querySelectorAll('.safety-radio');
                                        const safeContainer = document.getElementById('safe-conditions');
                                        const unsafeContainer = document.getElementById('unsafe-conditions');
                                        
                                        // Update threshold bars dynamically
                                        const thresholdInputs = document.querySelectorAll('.threshold-input');
                                        thresholdInputs.forEach(input => {
                                            input.addEventListener('input', function() {
                                                updateThresholdBar(this.dataset.condition, parseInt(this.value) || 0);
                                            });
                                        });
                                        
                                        radios.forEach(radio => {
                                            radio.addEventListener('change', function() {
                                                const condition = this.dataset.condition;
                                                const newSafety = this.value;
                                                const card = document.querySelector(`.condition-card[data-condition="${condition}"]`);
                                                
                                                if (card) {
                                                    // Update card data attribute
                                                    card.dataset.safety = newSafety;
                                                    
                                                    // Move card to appropriate section
                                                    if (newSafety === 'safe') {
                                                        safeContainer.appendChild(card);
                                                    } else {
                                                        unsafeContainer.appendChild(card);
                                                    }
                                                    
                                                    // Update threshold bar color to match new safety classification
                                                    const thresholdInput = card.querySelector('.threshold-input');
                                                    if (thresholdInput) {
                                                        updateThresholdBar(condition, parseInt(thresholdInput.value) || 0);
                                                    }
                                                    
                                                    // Add subtle animation
                                                    card.style.animation = 'none';
                                                    setTimeout(() => {
                                                        card.style.animation = 'slideIn 0.3s ease';
                                                    }, 10);
                                                }
                                            });
                                        });
                                    });
                                    
                                    // Update threshold bar visualization
                                    function updateThresholdBar(condition, value) {
                                        const barBelow = document.getElementById(`bar_below_${condition}`);
                                        const barAbove = document.getElementById(`bar_above_${condition}`);
                                        const barMarker = document.getElementById(`bar_marker_${condition}`);
                                        const barLabel = document.getElementById(`bar_label_${condition}`);
                                        
                                        if (!barBelow || !barAbove || !barMarker || !barLabel) return;
                                        
                                        // Clamp value between 0 and 100
                                        value = Math.max(0, Math.min(100, value));
                                        
                                        // Update widths
                                        barBelow.style.width = value + '%';
                                        barAbove.style.width = (100 - value) + '%';
                                        barMarker.style.left = value + '%';
                                        
                                        // Update marker label
                                        const markerLabel = barMarker.querySelector('div');
                                        if (markerLabel) {
                                            markerLabel.textContent = value + '%';
                                        }
                                        
                                        // Update trigger label
                                        barLabel.textContent = `✓ Triggers >${value}%`;
                                        
                                        // Update bar color based on current safety classification
                                        const card = document.querySelector(`.condition-card[data-condition="${condition}"]`);
                                        if (card) {
                                            const isSafe = card.dataset.safety === 'safe';
                                            barAbove.style.background = isSafe ? 'rgba(52, 211, 153, 0.6)' : 'rgba(248, 113, 113, 0.6)';
                                        }
                                    }
                                    </script>
                                    <style>
                                    @keyframes slideIn {
                                        from {
                                            opacity: 0;
                                            transform: translateX(-10px);
                                        }
                                        to {
                                            opacity: 1;
                                            transform: translateX(0);
                                        }
                                    }
                                    </style>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                                
                                <button type="submit">💾 Save Configuration</button>
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
    last_session_duration_str = None
    if safety_monitor.connected:
        ascom_status = 'Connected'
        ascom_status_class = 'status-connected'
        
        # Calculate connection duration
        if safety_monitor.connected_at:
            duration = get_current_time(safety_monitor.alpaca_config.timezone) - safety_monitor.connected_at
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
        if safety_monitor.last_session_duration is not None:
            duration = safety_monitor.last_session_duration
            hours, remainder = divmod(int(duration), 3600)
            minutes, seconds = divmod(remainder, 60)
            if hours > 0:
                last_session_duration_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                last_session_duration_str = f"{minutes}m {seconds}s"
            else:
                last_session_duration_str = f"{seconds}s"
    
    # Format last connected/disconnected timestamps with current timezone
    last_connected = None
    if safety_monitor.last_connected_at:
        # Convert to current timezone if different
        try:
            tz = ZoneInfo(safety_monitor.alpaca_config.timezone)
            converted_time = safety_monitor.last_connected_at.astimezone(tz)
            last_connected = converted_time.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            last_connected = safety_monitor.last_connected_at.strftime('%Y-%m-%d %H:%M:%S')
    
    last_disconnected = None
    if safety_monitor.disconnected_at:
        # Convert to current timezone if different
        try:
            tz = ZoneInfo(safety_monitor.alpaca_config.timezone)
            converted_time = safety_monitor.disconnected_at.astimezone(tz)
            last_disconnected = converted_time.strftime('%Y-%m-%d %H:%M:%S')
        except Exception:
            last_disconnected = safety_monitor.disconnected_at.strftime('%Y-%m-%d %H:%M:%S')
    
    # Prepare ASCOM safe/unsafe status display
    ascom_safe_status = 'SAFE' if safety_monitor._stable_safe_state else 'UNSAFE'
    ascom_safe_color = 'rgb(52, 211, 153)' if safety_monitor._stable_safe_state else 'rgb(248, 113, 113)'
    
    # Format safety history for display (newest first) with current timezone
    safety_history = []
    for entry in reversed(safety_monitor._safety_history):
        try:
            tz = ZoneInfo(safety_monitor.alpaca_config.timezone)
            converted_time = entry['timestamp'].astimezone(tz)
            time_str = converted_time.strftime('%H:%M:%S')
            date_str = converted_time.strftime('%Y-%m-%d')
        except Exception:
            time_str = entry['timestamp'].strftime('%H:%M:%S')
            date_str = entry['timestamp'].strftime('%Y-%m-%d')
        
        safety_history.append({
            'time': time_str,
            'date': date_str,
            'is_safe': entry['is_safe'],
            'status': 'SAFE' if entry['is_safe'] else 'UNSAFE',
            'condition': entry['condition'],
            'confidence': f"{entry['confidence']:.1f}"
        })
    
    uptime_delta = datetime.now() - start_time
    days = uptime_delta.days
    hours, rem = divmod(uptime_delta.seconds, 3600)
    minutes, seconds = divmod(rem, 60)

    if days > 0:
        container_uptime = f"{days}d {hours}h {minutes}m"
    elif hours > 0:
        container_uptime = f"{hours}h {minutes}m {seconds}s"
    else:
        container_uptime = f"{minutes}m {seconds}s"

    return render_template_string(
        html_template,
        message=message,
        current_name=safety_monitor.alpaca_config.device_name,
        current_location=safety_monitor.alpaca_config.location,
        current_image_url=safety_monitor.alpaca_config.image_url,
        current_ntp_server=safety_monitor.alpaca_config.ntp_server,
        current_timezone=safety_monitor.alpaca_config.timezone,
        image_url_default=os.environ.get('IMAGE_URL', 'Not set in environment'),
        all_conditions=all_available_conditions,
        unsafe_conditions=unsafe_conditions,
        safe_conditions=safe_conditions,
        current_condition=current_condition,
        current_confidence=current_confidence,
        detection_time=detection_time,
        last_update=last_update,
        ascom_status=ascom_status,
        ascom_status_class=ascom_status_class,
        ascom_safe_status=ascom_safe_status,
        ascom_safe_color=ascom_safe_color,
        safety_history=safety_history,
        connection_duration=connection_duration,
        last_session_duration=last_session_duration_str,
        client_ip=safety_monitor.client_ip,
        last_connected=last_connected,
        last_disconnected=last_disconnected,
        detection_interval=safety_monitor.alpaca_config.detection_interval,
        update_interval=safety_monitor.alpaca_config.update_interval,
        debounce_to_safe=safety_monitor.alpaca_config.debounce_to_safe_sec,
        debounce_to_unsafe=safety_monitor.alpaca_config.debounce_to_unsafe_sec,
        default_threshold=safety_monitor.alpaca_config.default_threshold,
        container_uptime=container_uptime,
        class_thresholds=safety_monitor.alpaca_config.class_thresholds
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
            detection_interval=int(os.getenv('DETECT_INTERVAL', '30')),
            update_interval=int(os.getenv('ALPACA_UPDATE_INTERVAL', '30'))
        )
        alpaca_config.save_to_file()
    else:
        # Override settings from environment if provided
        alpaca_config.port = int(os.getenv('ALPACA_PORT', str(alpaca_config.port)))
        alpaca_config.device_number = int(os.getenv('ALPACA_DEVICE_NUMBER', str(alpaca_config.device_number)))
        alpaca_config.detection_interval = int(os.getenv('DETECT_INTERVAL', str(alpaca_config.detection_interval)))
        alpaca_config.update_interval = int(os.getenv('ALPACA_UPDATE_INTERVAL', str(alpaca_config.update_interval)))
    
    # Sync detect_config interval with alpaca_config
    detect_config.detect_interval = alpaca_config.detection_interval
    
    # 2. Initialize global safety monitor
    safety_monitor = AlpacaSafetyMonitor(alpaca_config, detect_config)
    
    # 3. Start detection loop (internal, always running for MQTT/web UI)
    # This starts the detection thread but doesn't set "connected" status
    safety_monitor.stop_detection.clear()
    safety_monitor.detection_thread = threading.Thread(target=safety_monitor._detection_loop, daemon=True)
    safety_monitor.detection_thread.start()
    logger.info("Starting unified detection loop (ASCOM + MQTT)")
    
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
