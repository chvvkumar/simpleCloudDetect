"""
ASCOM Alpaca SafetyMonitor Device Implementation
"""
import threading
import logging
import io
import json
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple, List
import paho.mqtt.client as mqtt
from PIL import Image

# Watchdog constants
CLIENT_TIMEOUT_SECONDS = 300  # 5 Minutes

# Import from sibling modules
from .config import AlpacaConfig, get_current_time, ALL_CLOUD_CONDITIONS
# Assuming detect.py is in the root path or installed as a package
from detect import CloudDetector, Config as DetectConfig, HADiscoveryManager

logger = logging.getLogger(__name__)


class AlpacaSafetyMonitor:
    """ASCOM Alpaca SafetyMonitor implementation with optimized image handling"""
    
    def __init__(self, alpaca_config: AlpacaConfig, detect_config: DetectConfig):
        self.alpaca_config = alpaca_config
        self.detect_config = detect_config
        self.server_transaction_id = 0
        self.transaction_lock = threading.Lock()
        
        # Device state
        self.connected_clients: Dict[Tuple[str, int], datetime] = {}  # (IP, ClientID) -> Connection Start Time
        self.client_last_seen: Dict[Tuple[str, int], datetime] = {}   # (IP, ClientID) -> Last Heartbeat Time
        self.disconnected_clients: Dict[Tuple[str, int], Tuple[datetime, datetime]] = {}  # (IP, ClientID) -> (ConnectionTime, DisconnectionTime)
        self.connection_lock = threading.Lock()
        self.connecting = False
        self.connected_at: Optional[datetime] = None
        self.disconnected_at: Optional[datetime] = None
        self.last_connected_at: Optional[datetime] = None
        self.client_ip: Optional[str] = None
        self.last_session_duration: Optional[float] = None
        
        # Cloud detection state
        self.latest_detection: Optional[Dict[str, Any]] = {
            'class_name': 'Unknown',
            'confidence_score': 0.0,
            'Detection Time (Seconds)': 0.0,
            'timestamp': None
        }
        self._cached_is_safe = False
        self._unsafe_conditions_set = set(alpaca_config.unsafe_conditions)
        
        # Debounce state tracking
        self._stable_safe_state = False
        self._pending_safe_state: Optional[bool] = None
        self._state_change_start_time: Optional[datetime] = None
        
        # Safety state history (last 100 transitions)
        self._safety_history = deque(maxlen=100)
        
        # OPTIMIZATION: Store raw bytes instead of base64 string
        self.latest_image_bytes: Optional[bytes] = None
        
        self.detection_lock = threading.Lock()
        self.detection_thread: Optional[threading.Thread] = None
        self.stop_detection = threading.Event()
        
        # Setup MQTT client first
        self.mqtt_client = self._setup_mqtt()
        self.ha_discovery = None
        
        # Pre-load cloud detector at startup
        logger.info("Pre-loading ML model...")
        self.cloud_detector = CloudDetector(self.detect_config, mqtt_client=self.mqtt_client)
        logger.info("ML model loaded successfully")
        
        # Setup HA Discovery if enabled
        if self.mqtt_client and self.detect_config.mqtt_discovery_mode == 'homeassistant':
            self.ha_discovery = HADiscoveryManager(self.detect_config, self.mqtt_client)
            self.ha_discovery.publish_discovery_configs()
        
        # Blocking initial detection to ensure readiness
        logger.info("Performing initial detection (blocking)...")
        self._run_single_detection(initial=True)
        logger.info(f"Initialized {self.alpaca_config.device_name}")
    
    def _create_thumbnail_bytes(self, img: Image.Image) -> Optional[bytes]:
        """Create raw JPEG bytes for thumbnail (Optimized)"""
        try:
            img_copy = img.copy()
            img_copy.thumbnail((200, 200), Image.Resampling.LANCZOS)
            buffered = io.BytesIO()
            img_copy.save(buffered, format="JPEG", quality=85)
            return buffered.getvalue()  # Return raw bytes
        except Exception as e:
            logger.warning(f"Thumbnail failed: {e}")
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
            self._cached_is_safe = self._stable_safe_state
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
                        required_duration = self.alpaca_config.debounce_to_safe_sec
                    else:
                        required_duration = self.alpaca_config.debounce_to_unsafe_sec
                    
                    if elapsed_time >= required_duration:
                        # Debounce period complete - commit state change
                        self._stable_safe_state = is_safe_now
                        self._cached_is_safe = is_safe_now
                        self._pending_safe_state = None
                        self._state_change_start_time = None
                        
                        # Add to safety history
                        self._safety_history.append({
                            'timestamp': get_current_time(self.alpaca_config.timezone),
                            'is_safe': is_safe_now,
                            'condition': class_name,
                            'confidence': confidence
                        })
                        
                        logger.warning(f"SAFETY STATE CHANGED: {'SAFE' if is_safe_now else 'UNSAFE'} "
                                     f"(class={class_name}, confidence={confidence:.1f}%, "
                                     f"threshold={threshold:.1f}%, debounce={elapsed_time:.1f}s)")
    
    def _prune_stale_clients(self):
        """Remove clients that haven't been seen for CLIENT_TIMEOUT_SECONDS (assumes lock is held)"""
        now = get_current_time(self.alpaca_config.timezone)
        cutoff_time = now - timedelta(seconds=CLIENT_TIMEOUT_SECONDS)
        
        stale_clients = []
        # Check last_seen for staleness, not initial connection time
        for key, last_seen in list(self.client_last_seen.items()):
            if last_seen < cutoff_time:
                stale_clients.append(key)
        
        for key in stale_clients:
            client_ip, client_id = key
            
            # Retrieve original connection time for the record
            conn_time = self.connected_clients.get(key, now)
            
            # Move to disconnected list
            self.disconnected_clients[key] = (conn_time, now)
            
            # Remove from active tracking
            if key in self.connected_clients:
                del self.connected_clients[key]
            if key in self.client_last_seen:
                del self.client_last_seen[key]
                
            logger.warning(f"Watchdog: Pruned stale client {client_ip} (ID: {client_id}) - "
                          f"inactive for {(now - last_seen).total_seconds():.0f}s")
    
    def register_heartbeat(self, client_ip: str, client_id: int):
        """Update the last seen timestamp for a connected client"""
        with self.connection_lock:
            # Prune stale clients during heartbeat (periodic cleanup)
            self._prune_stale_clients()
            
            key = (client_ip, client_id)
            if key in self.connected_clients:
                # Only update last_seen, preserve connected_clients (start time)
                self.client_last_seen[key] = get_current_time(self.alpaca_config.timezone)
    
    def _setup_mqtt(self):
        """Setup and return MQTT client based on detect_config"""
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
    
    def _run_single_detection(self, initial: bool = False):
        """Run a single detection cycle with optimized image handling"""
        try:
            # Return image so we can process thumbnail
            result = self.cloud_detector.detect(return_image=True)
            result['timestamp'] = get_current_time(self.alpaca_config.timezone)
            
            # FIX: Normalize class name case to match configuration (Title Case preference)
            # This handles cases where detection returns "mostly cloudy" but config expects "Mostly Cloudy"
            raw_class = result.get('class_name', '')
            for known_cond in ALL_CLOUD_CONDITIONS:
                if known_cond.lower() == raw_class.lower():
                    result['class_name'] = known_cond
                    break
            
            # OPTIMIZATION: Convert to bytes immediately and drop the PIL Object
            image_bytes = None
            if 'image' in result:
                image_bytes = self._create_thumbnail_bytes(result['image'])
                del result['image']
            
            with self.detection_lock:
                self.latest_detection = result
                self.latest_image_bytes = image_bytes
                self._update_cached_safety(result)
                
                if initial:
                    self._safety_history.append({
                        'timestamp': result['timestamp'],
                        'is_safe': self._stable_safe_state,
                        'condition': result.get('class_name', 'Unknown'),
                        'confidence': result.get('confidence_score', 0.0)
                    })
            
            # MQTT Publish
            if self.mqtt_client:
                try:
                    mqtt_result = result.copy()
                    if 'timestamp' in mqtt_result and isinstance(mqtt_result['timestamp'], datetime):
                        mqtt_result['timestamp'] = mqtt_result['timestamp'].isoformat()
                    
                    if self.detect_config.mqtt_discovery_mode == 'homeassistant':
                        self.ha_discovery.publish_states(mqtt_result)
                    else:
                        self.mqtt_client.publish(self.detect_config.topic, json.dumps(mqtt_result))
                except Exception as e:
                    logger.error(f"MQTT publish failed: {e}")

        except Exception as e:
            logger.error(f"Detection cycle failed: {e}")
    
    def _detection_loop(self):
        """Background thread for continuous cloud detection"""
        logger.info("Starting detection loop")
        
        while not self.stop_detection.is_set():
            self._run_single_detection()
            logger.info(f"Cloud detection: {self.latest_detection['class_name']} "
                       f"({self.latest_detection['confidence_score']:.1f}%)")
            
            if self.stop_detection.wait(self.alpaca_config.update_interval):
                break
        
        logger.info("Detection loop stopped")
    
    def get_next_transaction_id(self) -> int:
        """Generate next server transaction ID (thread-safe, wraps at uint32 max)"""
        with self.transaction_lock:
            self.server_transaction_id = (self.server_transaction_id + 1) % 4294967296
            return self.server_transaction_id
    
    def create_response(self, value: Any = None, error_number: int = 0, 
                       error_message: str = "", client_transaction_id: int = 0) -> Dict[str, Any]:
        """Create standard ASCOM Alpaca response"""
        response = {
            "ClientTransactionID": client_transaction_id,
            "ServerTransactionID": self.get_next_transaction_id(),
            "ErrorNumber": error_number,
            "ErrorMessage": error_message
        }
        
        if value is not None or error_number == 0:
            response["Value"] = value
        
        return response
    
    def get_client_params(self) -> tuple:
        """Extract client ID and transaction ID from request with validation"""
        from flask import request
        client_id_raw = self._get_arg('ClientID', '0')
        client_tx_raw = self._get_arg('ClientTransactionID', '0')
        
        client_id_raw = str(client_id_raw).strip()
        client_tx_raw = str(client_tx_raw).strip()
        
        try:
            client_id = int(client_id_raw) if client_id_raw else 0
        except (ValueError, TypeError):
            logger.warning(f"Invalid ClientID received: {client_id_raw}")
            client_id = 0
        
        try:
            client_transaction_id = int(client_tx_raw) if client_tx_raw else 0
            if client_transaction_id < 0 or client_transaction_id > 4294967295:
                logger.warning(f"ClientTransactionID out of range: {client_transaction_id}")
                client_transaction_id = 0
        except (ValueError, TypeError):
            logger.warning(f"Invalid ClientTransactionID received: {client_tx_raw}")
            client_transaction_id = 0
            
        return client_id, client_transaction_id
    
    def _get_arg(self, key: str, default: Any = None) -> str:
        """Case-insensitive argument retrieval from request values"""
        from flask import request
        key_lower = key.lower()
        for k, v in request.values.items():
            if k.lower() == key_lower:
                return v
        return default
    
    @property
    def is_connected(self) -> bool:
        """Check if any clients are connected"""
        with self.connection_lock:
            return len(self.connected_clients) > 0
    
    def is_client_connected(self, client_ip: str, client_id: int) -> bool:
        """Check if a specific client is connected"""
        with self.connection_lock:
            key = (client_ip, client_id)
            return key in self.connected_clients

    def connect(self, client_ip: str, client_id: int):
        """Connect a client to the device"""
        with self.connection_lock:
            # Prune stale clients BEFORE adding new connection
            self._prune_stale_clients()
            
            key = (client_ip, client_id)
            current_time = get_current_time(self.alpaca_config.timezone)
            self.connected_clients[key] = current_time
            self.client_last_seen[key] = current_time
            
            # Remove from disconnected clients if reconnecting
            if key in self.disconnected_clients:
                del self.disconnected_clients[key]
            
            if len(self.connected_clients) == 1:
                self.connected_at = self.connected_clients[key]
                self.last_connected_at = self.connected_at
                self.disconnected_at = None
                
            logger.info(f"Client connected: {client_ip} (ID: {client_id}). Total clients: {len(self.connected_clients)}")

    def disconnect(self, client_ip: str = None, client_id: int = None):
        """Disconnect a client from the device"""
        with self.connection_lock:
            if client_ip is None or client_id is None:
                # Disconnect all - IMMEDIATE state change
                disc_time = get_current_time(self.alpaca_config.timezone)
                for key in list(self.connected_clients.keys()):
                    conn_time = self.connected_clients[key]
                    self.disconnected_clients[key] = (conn_time, disc_time)
                
                self.connected_clients.clear()
                self.client_last_seen.clear()
                self.disconnected_at = disc_time
                if self.connected_at:
                    duration = (self.disconnected_at - self.connected_at).total_seconds()
                    self.last_session_duration = duration
                self.connected_at = None
                logger.info("All clients disconnected")
            else:
                key = (client_ip, client_id)
                if key in self.connected_clients:
                    conn_time = self.connected_clients[key]
                    disc_time = get_current_time(self.alpaca_config.timezone)
                    self.disconnected_clients[key] = (conn_time, disc_time)
                    
                    del self.connected_clients[key]
                    if key in self.client_last_seen:
                        del self.client_last_seen[key]
                        
                    logger.info(f"Client disconnected: {client_ip} (ID: {client_id}). Total clients: {len(self.connected_clients)}")
                    
                    if len(self.connected_clients) == 0:
                        self.disconnected_at = disc_time
                        if self.connected_at:
                            duration = (self.disconnected_at - self.connected_at).total_seconds()
                            self.last_session_duration = duration
                        self.connected_at = None
                        logger.info("All clients disconnected")
                else:
                    logger.warning(f"Attempted to disconnect unknown client: {client_ip} (ID: {client_id})")

    def is_safe(self) -> bool:
        """Determine if conditions are safe based on latest detection"""
        # Safety Fail-safe: Always return False if not connected
        if not self.is_connected:
            return False
            
        with self.detection_lock:
            return self._stable_safe_state
    
    def get_pending_status(self) -> Dict[str, Any]:
        """Get information about any pending state changes"""
        with self.detection_lock:
            if self._pending_safe_state is None or self._state_change_start_time is None:
                return {'is_pending': False}
            
            now = get_current_time(self.alpaca_config.timezone)
            elapsed = (now - self._state_change_start_time).total_seconds()
            
            if self._pending_safe_state:
                required = self.alpaca_config.debounce_to_safe_sec
            else:
                required = self.alpaca_config.debounce_to_unsafe_sec
                
            remaining = max(0, required - elapsed)
            
            return {
                'is_pending': True,
                'target_state': 'SAFE' if self._pending_safe_state else 'UNSAFE',
                'target_color': 'rgb(52, 211, 153)' if self._pending_safe_state else 'rgb(248, 113, 113)',
                'remaining_seconds': round(remaining, 1),
                'total_duration': required
            }
    
    def get_safety_history(self) -> List[Dict[str, Any]]:
        """Get a thread-safe copy of the safety history"""
        with self.detection_lock:
            return list(self._safety_history)

    def get_connected_clients_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about connected clients"""
        clients = []
        now = get_current_time(self.alpaca_config.timezone)
        with self.connection_lock:
            # Prune stale clients before returning list
            self._prune_stale_clients()
            
            for (ip, client_id), conn_time in self.connected_clients.items():
                last_seen = self.client_last_seen.get((ip, client_id))
                clients.append({
                    "ip": ip,
                    "client_id": client_id,
                    "connected_at": conn_time,
                    "last_seen": last_seen,
                    "duration_seconds": (now - conn_time).total_seconds()
                })
        return clients

    def get_device_state(self) -> list:
        """Get current operational state"""
        is_safe_val = self.is_safe()
        return [{"Name": "IsSafe", "Value": is_safe_val}]
