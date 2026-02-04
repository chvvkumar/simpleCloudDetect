"""
External REST API Blueprint
Provides access to system status, configuration, and detection data for external integrations.
Separated from ASCOM Alpaca routes to maintain strict compliance there.
"""
import logging
from datetime import datetime
from flask import Blueprint, jsonify, Response
from ..device import AlpacaSafetyMonitor

logger = logging.getLogger(__name__)

external_api_bp = Blueprint('external_api', __name__)
monitor: AlpacaSafetyMonitor = None
start_time = datetime.now()

def init_external_api(safety_monitor_instance: AlpacaSafetyMonitor):
    """Initialize the external API blueprint with the safety monitor instance"""
    global monitor
    monitor = safety_monitor_instance

@external_api_bp.route('/api/ext/v1/system', methods=['GET'])
def get_system_info():
    """Get system information and uptime"""
    uptime = (datetime.now() - start_time).total_seconds()
    return jsonify({
        "name": "SimpleCloudDetect",
        "uptime_seconds": uptime,
        "uptime_formatted": str(datetime.now() - start_time).split('.')[0],
        "server_time": datetime.now().isoformat()
    })

@external_api_bp.route('/api/ext/v1/status', methods=['GET'])
def get_status():
    """Get current safety status and latest detection"""
    if not monitor:
        return jsonify({"error": "System not initialized"}), 503
        
    # Get safety status
    is_safe = monitor.is_safe()
    
    # Get latest detection safely
    detection = {}
    with monitor.detection_lock:
        if monitor.latest_detection:
            detection = monitor.latest_detection.copy()
            # Serialize timestamp
            if detection.get('timestamp'):
                detection['timestamp'] = detection['timestamp'].isoformat()
    
    return jsonify({
        "is_safe": is_safe,
        "safety_status": "Safe" if is_safe else "Unsafe",
        "detection": detection
    })

@external_api_bp.route('/api/ext/v1/config', methods=['GET'])
def get_config():
    """Get current configuration settings"""
    if not monitor:
        return jsonify({"error": "System not initialized"}), 503
        
    cfg = monitor.alpaca_config
    
    return jsonify({
        "device": {
            "name": cfg.device_name,
            "location": cfg.location,
            "id": cfg.device_number
        },
        "imaging": {
            "url": cfg.image_url,
            "interval": cfg.detection_interval
        },
        "safety": {
            "unsafe_conditions": cfg.unsafe_conditions,
            "thresholds": cfg.class_thresholds,
            "default_threshold": cfg.default_threshold,
            "debounce_safe_sec": cfg.debounce_to_safe_sec,
            "debounce_unsafe_sec": cfg.debounce_to_unsafe_sec
        },
        "system": {
            "timezone": cfg.timezone,
            "ntp_server": cfg.ntp_server,
            "update_interval": cfg.update_interval
        }
    })

@external_api_bp.route('/api/ext/v1/clients', methods=['GET'])
def get_clients():
    """Get connected ASCOM Alpaca clients"""
    if not monitor:
        return jsonify({"error": "System not initialized"}), 503
        
    client_list = monitor.get_connected_clients_info()
    
    # Serialize datetimes
    for client in client_list:
        if client.get('connected_at'):
            client['connected_at'] = client['connected_at'].isoformat()
        if client.get('last_seen'):
            client['last_seen'] = client['last_seen'].isoformat()
            
    return jsonify({
        "connected_count": len(client_list),
        "clients": client_list
    })

@external_api_bp.route('/api/ext/v1/history', methods=['GET'])
def get_history():
    """Get safety state transition history"""
    if not monitor:
        return jsonify({"error": "System not initialized"}), 503
        
    history = []
    raw_history = monitor.get_safety_history()
    
    for entry in raw_history:
        item = entry.copy()
        if item.get('timestamp'):
            item['timestamp'] = item['timestamp'].isoformat()
        history.append(item)
            
    # Return in reverse chronological order (newest first)
    return jsonify(list(reversed(history)))

@external_api_bp.route('/api/ext/v1/image', methods=['GET'])
def get_image():
    """Get the latest detection image (raw JPEG)"""
    if not monitor:
        return jsonify({"error": "System not initialized"}), 503
        
    if monitor.latest_image_bytes:
        return Response(monitor.latest_image_bytes, mimetype='image/jpeg')
    else:
        return jsonify({"error": "No image available"}), 404