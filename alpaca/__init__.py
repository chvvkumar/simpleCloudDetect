"""
Alpaca Package - ASCOM Alpaca SafetyMonitor
"""
import os
from flask import Flask
from flask_cors import CORS
from .config import AlpacaConfig
from .device import AlpacaSafetyMonitor
from .routes.api import api_bp, init_api
from .routes.management import mgmt_bp, init_mgmt
from detect import Config as DetectConfig

def create_app():
    """Flask application factory"""
    app = Flask(__name__, template_folder='../templates')
    CORS(app)
    
    # Load configs
    alpaca_cfg = AlpacaConfig.load_from_file()
    if alpaca_cfg is None:
        # Create default config from environment
        alpaca_cfg = AlpacaConfig(
            port=int(os.getenv('ALPACA_PORT', '11111')),
            device_number=int(os.getenv('ALPACA_DEVICE_NUMBER', '0')),
            detection_interval=int(os.getenv('DETECT_INTERVAL', '30')),
            update_interval=int(os.getenv('ALPACA_UPDATE_INTERVAL', '30'))
        )
        alpaca_cfg.save_to_file()
    else:
        # Override settings from environment if provided
        alpaca_cfg.port = int(os.getenv('ALPACA_PORT', str(alpaca_cfg.port)))
        alpaca_cfg.device_number = int(os.getenv('ALPACA_DEVICE_NUMBER', str(alpaca_cfg.device_number)))
        alpaca_cfg.detection_interval = int(os.getenv('DETECT_INTERVAL', str(alpaca_cfg.detection_interval)))
        alpaca_cfg.update_interval = int(os.getenv('ALPACA_UPDATE_INTERVAL', str(alpaca_cfg.update_interval)))
    
    detect_cfg = DetectConfig.from_env()
    
    # Sync detect_config interval with alpaca_config
    detect_cfg.detect_interval = alpaca_cfg.detection_interval
    
    # Initialize Core Device
    safety_monitor = AlpacaSafetyMonitor(alpaca_cfg, detect_cfg)
    
    # Initialize Routes with Monitor Instance
    init_api(safety_monitor)
    init_mgmt(safety_monitor)
    
    # Register Blueprints
    app.register_blueprint(api_bp, url_prefix='/api')
    app.register_blueprint(mgmt_bp, url_prefix='')
    
    # Store monitor for access in main.py
    app.safety_monitor = safety_monitor
    
    return app, safety_monitor
