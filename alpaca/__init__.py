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
    
    # 1. Initialize configuration from Environment Variables (Base Config)
    # This establishes the defaults if no file exists.
    alpaca_cfg = AlpacaConfig(
        port=int(os.getenv('ALPACA_PORT', '11111')),
        device_number=int(os.getenv('ALPACA_DEVICE_NUMBER', '0')),
        detection_interval=int(os.getenv('DETECT_INTERVAL', '30')),
        update_interval=int(os.getenv('ALPACA_UPDATE_INTERVAL', '30'))
    )

    # 2. Load from file and override environment settings if file exists
    # This ensures user settings (in file) take precedence over preconfigured env vars,
    # but Env vars are still preserved if the file doesn't specify them.
    file_settings = AlpacaConfig.load_settings_from_file()
    if file_settings:
        # Update our base config with ONLY the values explicitly in the file
        for key, value in file_settings.items():
            setattr(alpaca_cfg, key, value)
    
    # 3. Save the final configuration to ensure the file exists and is current
    alpaca_cfg.save_to_file()
    
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
