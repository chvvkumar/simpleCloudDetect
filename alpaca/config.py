"""
Configuration management for the Alpaca server
"""
import os
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Dict
from zoneinfo import ZoneInfo
from datetime import datetime

logger = logging.getLogger(__name__)

# Global helper
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

# Dynamically load cloud conditions from labels file
def load_labels():
    """Load cloud condition labels from labels.txt file"""
    label_path = os.environ.get('LABEL_PATH', 'labels.txt')
    try:
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                labels = []
                for line in f:
                    clean_line = line.strip()
                    # Handle "0 Clear" format if present
                    if " " in clean_line and clean_line.split(" ", 1)[0].isdigit():
                        clean_line = clean_line.split(" ", 1)[1]
                    if clean_line:
                        labels.append(clean_line)
            logger.info(f"Loaded {len(labels)} classes from {label_path}")
            return labels
    except Exception as e:
        logger.error(f"Failed to load labels from {label_path}: {e}")
    
    # Fallback to defaults if file missing or error
    logger.warning("Using fallback default cloud conditions")
    return ['Clear', 'Mostly Cloudy', 'Overcast', 'Rain', 'Snow', 'Wisps of clouds']

# Available cloud conditions from ML model (loaded dynamically from labels.txt)
ALL_CLOUD_CONDITIONS = load_labels()


@dataclass
class AlpacaConfig:
    """Configuration for the Alpaca server"""
    port: int = 11111
    device_number: int = 0
    device_name: str = "SimpleCloudDetect"
    device_description: str = "ASCOM SafetyMonitor based on ML cloud detection"
    driver_info: str = "ASCOM Alpaca SafetyMonitor v2.0 - Cloud Detection Driver"
    driver_version: str = "2.0"
    interface_version: int = 3
    detection_interval: int = 30  # seconds between ML detections (from detect.py)
    update_interval: int = 30  # seconds between cloud detection updates
    location: str = "AllSky Camera"
    image_url: str = field(default_factory=lambda: os.environ.get('IMAGE_URL', ''))
    unsafe_conditions: list = field(default_factory=lambda: ALL_CLOUD_CONDITIONS.copy())
    
    # Confidence threshold settings
    default_threshold: float = 50.0  # Default threshold for any class not explicitly configured
    class_thresholds: Dict[str, float] = field(default_factory=dict)  # Map class names to thresholds
    
    # Debounce settings (in seconds)
    debounce_to_safe_sec: int = 60  # Wait time before switching from Unsafe → Safe
    debounce_to_unsafe_sec: int = 0  # Wait time before switching from Safe → Unsafe (immediate)
    
    # NTP and timezone settings
    ntp_server: str = field(default_factory=lambda: os.environ.get('NTP_SERVER', 'pool.ntp.org'))
    timezone: str = field(default_factory=lambda: os.environ.get('TZ', 'UTC'))
    
    @classmethod
    def get_config_path(cls) -> str:
        """Get configuration file path from environment or default"""
        return os.environ.get('CONFIG_FILE', 'alpaca_config.json')

    def save_to_file(self):
        """Save configuration to JSON file"""
        filepath = self.get_config_path()
        try:
            # Create directory if it doesn't exist (for /config volume usage)
            dir_path = os.path.dirname(os.path.abspath(filepath))
            os.makedirs(dir_path, exist_ok=True)
            
            config_dict = asdict(self)
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
            logger.info(f"Configuration saved to {filepath}")
        except PermissionError:
            # Enhanced error logging for permission issues
            try:
                dir_path = os.path.dirname(os.path.abspath(filepath))
                stat_info = os.stat(dir_path)
                logger.error(f"Permission denied saving to {filepath}. "
                             f"Directory '{dir_path}' is owned by UID {stat_info.st_uid} with mode {oct(stat_info.st_mode)[-3:]}. "
                             f"Container running as UID {os.getuid()}. "
                             f"Fix with: sudo chown {os.getuid()} {dir_path}")
            except Exception:
                logger.error(f"Failed to save configuration to {filepath}: [Errno 13] Permission denied")
        except Exception as e:
            logger.error(f"Failed to save configuration to {filepath}: {e}")
    
    @classmethod
    def load_settings_from_file(cls) -> dict:
        """Load configuration dictionary from JSON file, filtering for valid fields"""
        filepath = cls.get_config_path()
        try:
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    config_dict = json.load(f)
                
                # Filter dictionary to only include valid fields for this dataclass
                valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
                settings = {k: v for k, v in config_dict.items() if k in valid_fields}

                # Sanitize loaded settings against current labels from labels.txt
                # Remove saved 'unsafe_conditions' that are no longer in ALL_CLOUD_CONDITIONS
                if 'unsafe_conditions' in settings:
                    valid_labels = set(ALL_CLOUD_CONDITIONS)
                    original_count = len(settings['unsafe_conditions'])
                    settings['unsafe_conditions'] = [
                        c for c in settings['unsafe_conditions'] 
                        if c in valid_labels
                    ]
                    removed_count = original_count - len(settings['unsafe_conditions'])
                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} obsolete unsafe condition(s) from saved configuration")
                
                # Clean up stale class thresholds
                if 'class_thresholds' in settings:
                    original_count = len(settings['class_thresholds'])
                    settings['class_thresholds'] = {
                        k: v for k, v in settings['class_thresholds'].items()
                        if k in ALL_CLOUD_CONDITIONS
                    }
                    removed_count = original_count - len(settings['class_thresholds'])
                    if removed_count > 0:
                        logger.info(f"Removed {removed_count} obsolete class threshold(s) from saved configuration")

                return settings
        except Exception as e:
            logger.error(f"Failed to load configuration from {filepath}: {e}")
        return {}

    @classmethod
    def load_from_file(cls):
        """Load configuration from JSON file with backward compatibility"""
        settings = cls.load_settings_from_file()
        if settings:
            logger.info(f"Configuration loaded from {cls.get_config_path()}")
            return cls(**settings)
        return None
