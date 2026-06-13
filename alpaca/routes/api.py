"""
ASCOM API Routes Blueprint
"""
from flask import Blueprint, jsonify, Response, request
import logging
from ..config import ERROR_SUCCESS, ERROR_INVALID_VALUE, ERROR_UNSPECIFIED
from ..device import AlpacaSafetyMonitor

logger = logging.getLogger(__name__)

# Create Blueprint
api_bp = Blueprint('api', __name__)

# Global monitor instance (injected by app factory)
monitor: AlpacaSafetyMonitor = None 

def init_api(safety_monitor_instance):
    """Initialize the API blueprint with the safety monitor instance"""
    global monitor
    monitor = safety_monitor_instance

def validate_device_number(device_number: int):
    """Validate device number and return error response if invalid"""
    if device_number != monitor.alpaca_config.device_number:
        _, client_tx_id = monitor.get_client_params()
        return jsonify(monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid device number: {device_number}",
            client_transaction_id=client_tx_id
        )), 400
    
    # Register heartbeat for watchdog (every API request keeps session alive)
    try:
        client_id, _ = monitor.get_client_params()
        client_ip = request.remote_addr
        monitor.register_heartbeat(client_ip, client_id)
    except Exception as e:
        # Don't fail the request if heartbeat fails
        logger.debug(f"Heartbeat registration failed: {e}")
    
    return None

def create_simple_get_endpoint(attribute_getter):
    """Factory for simple GET endpoints that return a config value"""
    def endpoint(device_number: int):
        error_response = validate_device_number(device_number)
        if error_response:
            return error_response
        _, client_tx_id = monitor.get_client_params()
        return jsonify(monitor.create_response(
            value=attribute_getter(),
            client_transaction_id=client_tx_id
        ))
    return endpoint

@api_bp.route('/v1/safetymonitor/<int:device_number>/issafe', methods=['GET'])
def get_issafe(device_number: int):
    """Get safety status"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    _, client_tx_id = monitor.get_client_params()
    return jsonify(monitor.create_response(
        value=monitor.is_safe(),
        client_transaction_id=client_tx_id
    ))

@api_bp.route('/v1/latest_image', methods=['GET'])
def get_latest_image():
    """Serve the latest detection image directly from memory bytes (OPTIMIZED)"""
    with monitor.detection_lock:
        img_bytes = monitor.latest_image_bytes
    
    if img_bytes:
        # OPTIMIZATION: Serve bytes directly without base64 encoding/decoding
        return Response(img_bytes, mimetype='image/jpeg')
    return Response(status=404)

@api_bp.route('/v1/safetymonitor/<int:device_number>/connected', methods=['GET'])
def get_connected(device_number: int):
    """Get connection state for this specific client"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    client_id, client_tx_id = monitor.get_client_params()
    client_ip = request.remote_addr
    # Return per-client connection state, not global state
    is_client_connected = monitor.is_client_connected(client_ip, client_id)
    return jsonify(monitor.create_response(
        value=is_client_connected,
        client_transaction_id=client_tx_id
    ))

@api_bp.route('/v1/safetymonitor/<int:device_number>/connected', methods=['PUT'])
def set_connected(device_number: int):
    """Set connection state"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    client_id, client_tx_id = monitor.get_client_params()
    connected_str = monitor._get_arg('Connected', '').strip().lower()
    
    if connected_str == 'true':
        target_state = True
    elif connected_str == 'false':
        target_state = False
    else:
        return jsonify(monitor.create_response(
            error_number=ERROR_INVALID_VALUE,
            error_message=f"Invalid boolean value for Connected: '{connected_str}'",
            client_transaction_id=client_tx_id
        )), 400

    try:
        client_ip = request.remote_addr
        if target_state:
            monitor.connect(client_ip=client_ip, client_id=client_id)
        else:
            monitor.disconnect(client_ip=client_ip, client_id=client_id)
        
        return jsonify(monitor.create_response(client_transaction_id=client_tx_id))
    except Exception as e:
        logger.error(f"Failed to set connected state: {e}")
        return jsonify(monitor.create_response(
            error_number=ERROR_UNSPECIFIED,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500

@api_bp.route('/v1/safetymonitor/<int:device_number>/connecting', methods=['GET'])
def get_connecting(device_number: int):
    """Get connecting state"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    _, client_tx_id = monitor.get_client_params()
    return jsonify(monitor.create_response(
        value=monitor.connecting,
        client_transaction_id=client_tx_id
    ))

@api_bp.route('/v1/safetymonitor/<int:device_number>/connect', methods=['PUT'])
def connect_device(device_number: int):
    """Connect to device"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    client_id, client_tx_id = monitor.get_client_params()
    try:
        client_ip = request.remote_addr
        monitor.connect(client_ip=client_ip, client_id=client_id)
        return jsonify(monitor.create_response(client_transaction_id=client_tx_id))
    except Exception as e:
        logger.error(f"Failed to connect: {e}")
        return jsonify(monitor.create_response(
            error_number=ERROR_UNSPECIFIED,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500

@api_bp.route('/v1/safetymonitor/<int:device_number>/disconnect', methods=['PUT'])
def disconnect_device(device_number: int):
    """Disconnect from device"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    
    client_id, client_tx_id = monitor.get_client_params()
    try:
        client_ip = request.remote_addr
        monitor.disconnect(client_ip=client_ip, client_id=client_id)
        return jsonify(monitor.create_response(client_transaction_id=client_tx_id))
    except Exception as e:
        logger.error(f"Error during disconnect: {e}")
        return jsonify(monitor.create_response(
            error_number=ERROR_UNSPECIFIED,
            error_message=str(e),
            client_transaction_id=client_tx_id
        )), 500

@api_bp.route('/v1/safetymonitor/<int:device_number>/description', methods=['GET'])
def get_description(device_number: int):
    return create_simple_get_endpoint(lambda: monitor.alpaca_config.device_description)(device_number)

@api_bp.route('/v1/safetymonitor/<int:device_number>/devicestate', methods=['GET'])
def get_devicestate(device_number: int):
    return create_simple_get_endpoint(monitor.get_device_state)(device_number)

@api_bp.route('/v1/safetymonitor/<int:device_number>/driverinfo', methods=['GET'])
def get_driverinfo(device_number: int):
    return create_simple_get_endpoint(lambda: monitor.alpaca_config.driver_info)(device_number)

@api_bp.route('/v1/safetymonitor/<int:device_number>/driverversion', methods=['GET'])
def get_driverversion(device_number: int):
    return create_simple_get_endpoint(lambda: monitor.alpaca_config.driver_version)(device_number)

@api_bp.route('/v1/safetymonitor/<int:device_number>/interfaceversion', methods=['GET'])
def get_interfaceversion(device_number: int):
    return create_simple_get_endpoint(lambda: monitor.alpaca_config.interface_version)(device_number)

@api_bp.route('/v1/safetymonitor/<int:device_number>/name', methods=['GET'])
def get_name(device_number: int):
    return create_simple_get_endpoint(lambda: monitor.alpaca_config.device_name)(device_number)

@api_bp.route('/v1/safetymonitor/<int:device_number>/supportedactions', methods=['GET'])
def get_supportedactions(device_number: int):
    return create_simple_get_endpoint(lambda: [])(device_number)

@api_bp.route('/v1/safetymonitor/<int:device_number>/action', methods=['PUT'])
def put_action(device_number: int):
    """Execute a device action"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    _, client_tx_id = monitor.get_client_params()
    return jsonify(monitor.create_response(
        error_number=ERROR_INVALID_VALUE,
        error_message="No actions are supported",
        client_transaction_id=client_tx_id
    )), 400

@api_bp.route('/v1/safetymonitor/<int:device_number>/commandblind', methods=['PUT'])
@api_bp.route('/v1/safetymonitor/<int:device_number>/commandbool', methods=['PUT'])
@api_bp.route('/v1/safetymonitor/<int:device_number>/commandstring', methods=['PUT'])
def not_implemented(device_number: int):
    """Not implemented commands"""
    error_response = validate_device_number(device_number)
    if error_response:
        return error_response
    _, client_tx_id = monitor.get_client_params()
    return jsonify(monitor.create_response(
        error_number=ERROR_INVALID_VALUE,
        error_message="Command not implemented",
        client_transaction_id=client_tx_id
    )), 400
