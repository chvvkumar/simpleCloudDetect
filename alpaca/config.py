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
    interface_version: int = 3
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
