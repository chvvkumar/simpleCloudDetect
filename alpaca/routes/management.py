"""
Management/Setup Routes Blueprint
"""
import os
from flask import Blueprint, render_template, request, redirect, url_for
from datetime import datetime
from zoneinfo import ZoneInfo
import logging
from ..device import AlpacaSafetyMonitor
from ..config import ALL_CLOUD_CONDITIONS, get_current_time

logger = logging.getLogger(__name__)

mgmt_bp = Blueprint('management', __name__)
monitor: AlpacaSafetyMonitor = None
start_time = datetime.now()

def init_mgmt(safety_monitor_instance):
    """Initialize the management blueprint with the safety monitor instance"""
    global monitor
    monitor = safety_monitor_instance

@mgmt_bp.route('/setup/v1/safetymonitor/<int:device_number>/setup', methods=['GET', 'POST'])
def setup_device(device_number: int):
    """Setup page for device configuration"""
    message = None
    
    if request.method == 'POST':
        try:
            # Update basic settings
            monitor.alpaca_config.device_name = request.form.get('device_name', monitor.alpaca_config.device_name)
            monitor.alpaca_config.location = request.form.get('location', monitor.alpaca_config.location)
            monitor.alpaca_config.image_url = request.form.get('image_url', monitor.alpaca_config.image_url)
            monitor.alpaca_config.ntp_server = request.form.get('ntp_server', monitor.alpaca_config.ntp_server)
            monitor.alpaca_config.timezone = request.form.get('timezone', monitor.alpaca_config.timezone)
            
            # Update timing settings
            monitor.alpaca_config.detection_interval = int(request.form.get('detection_interval', monitor.alpaca_config.detection_interval))
            monitor.alpaca_config.update_interval = int(request.form.get('update_interval', monitor.alpaca_config.update_interval))
            monitor.alpaca_config.debounce_to_safe_sec = int(request.form.get('debounce_safe', monitor.alpaca_config.debounce_to_safe_sec))
            monitor.alpaca_config.debounce_to_unsafe_sec = int(request.form.get('debounce_unsafe', monitor.alpaca_config.debounce_to_unsafe_sec))
            
            # Update unsafe conditions based on radio buttons
            new_unsafe = []
            for condition in ALL_CLOUD_CONDITIONS:
                safety_value = request.form.get(f'safety_{condition}')
                if safety_value == 'unsafe':
                    new_unsafe.append(condition)
            monitor.alpaca_config.unsafe_conditions = new_unsafe
            monitor._unsafe_conditions_set = set(new_unsafe)
            
            # Update thresholds
            new_thresholds = {}
            for condition in ALL_CLOUD_CONDITIONS:
                threshold_value = request.form.get(f'threshold_{condition}')
                if threshold_value:
                    new_thresholds[condition] = float(threshold_value)
            monitor.alpaca_config.class_thresholds = new_thresholds
            
            # Save configuration
            monitor.alpaca_config.save_to_file()
            message = "Configuration saved successfully!"
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
            message = f"Error saving configuration: {e}"
        
        return redirect(url_for('management.setup_device', device_number=device_number))
    
    # Prepare context for template rendering
    with monitor.detection_lock:
        det = monitor.latest_detection
        current_condition = det.get('class_name', 'Unknown')
        current_confidence = round(det.get('confidence_score', 0.0), 1)
        detection_time = round(det.get('Detection Time (Seconds)', 0.0), 2)
        timestamp = det.get('timestamp')
    
    # Calculate safe vs unsafe conditions
    unsafe_cond = monitor.alpaca_config.unsafe_conditions
    safe_cond = [c for c in ALL_CLOUD_CONDITIONS if c not in unsafe_cond]
    
    # ASCOM status
    is_safe = monitor.is_safe()
    ascom_safe_status = "SAFE" if is_safe else "UNSAFE"
    ascom_safe_color = "rgb(52, 211, 153)" if is_safe else "rgb(248, 113, 113)"
    
    # Format timestamp
    if timestamp:
        last_update = timestamp.strftime("%H:%M:%S")
    else:
        last_update = "N/A"
    
    # Container uptime
    uptime_seconds = (datetime.now() - start_time).total_seconds()
    uptime_hours = int(uptime_seconds // 3600)
    uptime_mins = int((uptime_seconds % 3600) // 60)
    container_uptime = f"{uptime_hours}h {uptime_mins}m"
    
    # Connection status
    ascom_status = "Connected" if monitor.is_connected else "Disconnected"
    ascom_status_class = "status-connected" if monitor.is_connected else "status-disconnected"
    client_count = len(monitor.connected_clients)
    
    # Build client list - show unique clients by IP with most recent connection info
    client_list = []
    with monitor.connection_lock:
        # Dictionary to track unique clients by IP
        unique_clients = {}
        
        # Process connected clients first (they take priority)
        for (ip, client_id), conn_time in monitor.connected_clients.items():
            if ip not in unique_clients or conn_time > unique_clients[ip]['conn_time']:
                duration = (get_current_time(monitor.alpaca_config.timezone) - conn_time).total_seconds()
                
                # Format duration as dd-hh-mm-ss fixed width
                dur_int = int(duration)
                days = dur_int // 86400
                hours = (dur_int % 86400) // 3600
                minutes = (dur_int % 3600) // 60
                seconds = dur_int % 60
                duration_str = f"{days:02d}d {hours:02d}h {minutes:02d}m {seconds:02d}s"

                try:
                    # Convert timestamp to current timezone
                    tz = ZoneInfo(monitor.alpaca_config.timezone)
                    local_conn_time = conn_time.astimezone(tz)
                    conn_time_str = local_conn_time.strftime("%Y-%m-%d %H:%M:%S")
                    conn_ts = local_conn_time.timestamp()
                except Exception:
                    # Fallback if timezone conversion fails
                    conn_time_str = conn_time.strftime("%Y-%m-%d %H:%M:%S")
                    conn_ts = conn_time.timestamp()
                
                unique_clients[ip] = {
                    'ip': ip,
                    'status': 'connected',
                    'duration': duration_str,
                    'duration_seconds': duration,
                    'connected_time': conn_time_str,
                    'connected_ts': conn_ts,
                    'disconnected_time': '-',
                    'disconnected_ts': 0,
                    'conn_time': conn_time  # Track for comparison
                }
        
        # Process disconnected clients only if IP not currently connected
        for (ip, client_id), (conn_time, disc_time) in monitor.disconnected_clients.items():
            # Skip if IP is currently connected
            if ip in unique_clients and unique_clients[ip]['status'] == 'connected':
                continue
            
            # Only add or update if this is the most recent disconnection for this IP
            if ip not in unique_clients or disc_time > unique_clients[ip].get('disc_time', datetime.min.replace(tzinfo=conn_time.tzinfo)):
                duration = (disc_time - conn_time).total_seconds()
                
                # Format duration as dd-hh-mm-ss fixed width
                dur_int = int(duration)
                days = dur_int // 86400
                hours = (dur_int % 86400) // 3600
                minutes = (dur_int % 3600) // 60
                seconds = dur_int % 60
                duration_str = f"{days:02d}d {hours:02d}h {minutes:02d}m {seconds:02d}s"

                try:
                    # Convert timestamps to current timezone
                    tz = ZoneInfo(monitor.alpaca_config.timezone)
                    local_conn_time = conn_time.astimezone(tz)
                    local_disc_time = disc_time.astimezone(tz)
                    conn_time_str = local_conn_time.strftime("%Y-%m-%d %H:%M:%S")
                    disc_time_str = local_disc_time.strftime("%Y-%m-%d %H:%M:%S")
                    conn_ts = local_conn_time.timestamp()
                    disc_ts = local_disc_time.timestamp()
                except Exception:
                    # Fallback if timezone conversion fails
                    conn_time_str = conn_time.strftime("%Y-%m-%d %H:%M:%S")
                    disc_time_str = disc_time.strftime("%Y-%m-%d %H:%M:%S")
                    conn_ts = conn_time.timestamp()
                    disc_ts = disc_time.timestamp()
                
                unique_clients[ip] = {
                    'ip': ip,
                    'status': 'disconnected',
                    'duration': duration_str,
                    'duration_seconds': duration,
                    'connected_time': conn_time_str,
                    'connected_ts': conn_ts,
                    'disconnected_time': disc_time_str,
                    'disconnected_ts': disc_ts,
                    'disc_time': disc_time  # Track for comparison
                }
        
        # Convert to list
        client_list = list(unique_clients.values())
    
    # Safety history (newest first - reverse chronological)
    safety_history = []
    with monitor.detection_lock:
        for entry in reversed(list(monitor._safety_history)[-10:]):  # Last 10 entries, newest first
            try:
                # Convert timestamp to current timezone
                tz = ZoneInfo(monitor.alpaca_config.timezone)
                converted_time = entry['timestamp'].astimezone(tz)
                time_str = converted_time.strftime("%H:%M:%S")
            except Exception:
                # Fallback if timezone conversion fails
                time_str = entry['timestamp'].strftime("%H:%M:%S")
            
            safety_history.append({
                'is_safe': entry['is_safe'],
                'time': time_str,
                'condition': entry['condition'],
                'confidence': round(entry['confidence'], 1)
            })
    
    # Render template
    return render_template(
        'setup.html',
        message=message,
        current_name=monitor.alpaca_config.device_name,
        current_location=monitor.alpaca_config.location,
        current_image_url=monitor.alpaca_config.image_url,
        image_url_default=os.environ.get('IMAGE_URL', ''),
        current_ntp_server=monitor.alpaca_config.ntp_server,
        current_timezone=monitor.alpaca_config.timezone,
        detection_interval=monitor.alpaca_config.detection_interval,
        update_interval=monitor.alpaca_config.update_interval,
        debounce_to_safe=monitor.alpaca_config.debounce_to_safe_sec,
        debounce_to_unsafe=monitor.alpaca_config.debounce_to_unsafe_sec,
        current_condition=current_condition,
        current_confidence=current_confidence,
        detection_time=detection_time,
        last_update=last_update,
        container_uptime=container_uptime,
        ascom_safe_status=ascom_safe_status,
        ascom_safe_color=ascom_safe_color,
        ascom_status=ascom_status,
        ascom_status_class=ascom_status_class,
        client_count=client_count,
        client_list=client_list,
        safety_history=safety_history,
        safe_conditions=safe_cond,
        unsafe_conditions=unsafe_cond,
        default_threshold=monitor.alpaca_config.default_threshold,
        class_thresholds=monitor.alpaca_config.class_thresholds
    )


# ASCOM Alpaca Management API Endpoints (must be at root, not under /api prefix)
@mgmt_bp.route('/management/apiversions', methods=['GET'])
def get_apiversions():
    """Get supported API versions"""
    return {
        "Value": [1], 
        "ErrorNumber": 0, 
        "ErrorMessage": ""
    }

@mgmt_bp.route('/management/v1/description', methods=['GET'])
def get_management_description():
    """Get server description"""
    return {
        "Value": {
            "ServerName": "SimpleCloudDetect",
            "Manufacturer": "SimpleCloudDetect",
            "ManufacturerVersion": "2.0",
            "Location": monitor.alpaca_config.location
        },
        "ErrorNumber": 0,
        "ErrorMessage": ""
    }

@mgmt_bp.route('/management/v1/configureddevices', methods=['GET'])
def get_configureddevices():
    """Get list of configured devices"""
    return {
        "Value": [{
            "DeviceName": monitor.alpaca_config.device_name,
            "DeviceType": "SafetyMonitor",
            "DeviceNumber": monitor.alpaca_config.device_number,
            "UniqueID": f"simpleclouddetect-safetymonitor-{monitor.alpaca_config.device_number}"
        }],
        "ErrorNumber": 0,
        "ErrorMessage": ""
    }
