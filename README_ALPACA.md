# ASCOM Alpaca SafetyMonitor

This is an ASCOM Alpaca-compliant SafetyMonitor server that uses machine learning-based cloud detection to determine observatory safety conditions.

## Overview

The SafetyMonitor integrates with the existing cloud detection system to provide real-time safety status via the ASCOM Alpaca REST API. It determines safety based on detected cloud conditions:

- **SAFE**: Clear, Wisps of clouds
- **UNSAFE**: Rain, Snow, Mostly Cloudy, Overcast

## Features

- ✅ Full ASCOM Alpaca API compliance (ISafetyMonitorV1)
- ✅ RESTful HTTP interface on port 11111 (configurable)
- ✅ **Automatic device discovery via UDP (port 32227)**
- ✅ Continuous cloud monitoring with configurable update intervals
- ✅ Thread-safe operations
- ✅ Comprehensive error handling
- ✅ Transaction ID tracking for debugging
- ✅ Platform 7 compatibility (DeviceState, Connecting)

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Ensure you have the required files:
   - `keras_model.h5` - Trained cloud detection model
   - `labels.txt` - Class labels
   - Camera image source (URL or file)

## Configuration

Configure via environment variables:

### Required (from existing cloud detection):
```bash
export IMAGE_URL="http://your-camera/image.jpg"
export MQTT_BROKER="your-mqtt-broker"
export MQTT_TOPIC="cloud/detection"
export DETECT_INTERVAL="60"
```

### Optional Alpaca-specific:
```bash
export ALPACA_PORT="11111"              # Default: 11111
export ALPACA_DEVICE_NUMBER="0"         # Default: 0
export ALPACA_UPDATE_INTERVAL="30"      # Seconds between updates, Default: 30
```

### Optional (with defaults):
```bash
export MODEL_PATH="keras_model.h5"      # Default: keras_model.h5
export LABEL_PATH="labels.txt"          # Default: labels.txt
export MQTT_PORT="1883"                 # Default: 1883
export MQTT_USERNAME=""                 # Optional
export MQTT_PASSWORD=""                 # Optional
```

## Usage

### Start the Server

```bash
python alpaca_safety_monitor.py
```

The server will start on port 11111 (or your configured port) and begin monitoring cloud conditions.

### Connect to the Device

Before querying safety status, connect to the device:

```bash
curl -X PUT "http://localhost:11111/api/v1/safetymonitor/0/connected" \
  -d "Connected=true"
```

### Query Safety Status

```bash
curl "http://localhost:11111/api/v1/safetymonitor/0/issafe"
```

Response:
```json
{
  "Value": true,
  "ClientTransactionID": 0,
  "ServerTransactionID": 1,
  "ErrorNumber": 0,
  "ErrorMessage": ""
}
```

### Get Device State (Platform 7)

```bash
curl "http://localhost:11111/api/v1/safetymonitor/0/devicestate"
```

Response:
```json
{
  "Value": [
    {"Name": "IsSafe", "Value": true},
    {"Name": "CloudCondition", "Value": "Clear"},
    {"Name": "Confidence", "Value": 98.5},
    {"Name": "LastUpdate", "Value": "2024-11-30T18:15:00Z"}
  ],
  "ClientTransactionID": 0,
  "ServerTransactionID": 2,
  "ErrorNumber": 0,
  "ErrorMessage": ""
}
```

## API Endpoints

### SafetyMonitor Specific

- **GET** `/api/v1/safetymonitor/0/issafe` - Returns safety status (true/false)

### Common Device Endpoints

- **GET/PUT** `/api/v1/safetymonitor/0/connected` - Connection state
- **GET** `/api/v1/safetymonitor/0/connecting` - Async connection status
- **GET** `/api/v1/safetymonitor/0/description` - Device description
- **GET** `/api/v1/safetymonitor/0/devicestate` - Operational state (Platform 7)
- **GET** `/api/v1/safetymonitor/0/driverinfo` - Driver information
- **GET** `/api/v1/safetymonitor/0/driverversion` - Driver version
- **GET** `/api/v1/safetymonitor/0/interfaceversion` - Interface version (returns 1)
- **GET** `/api/v1/safetymonitor/0/name` - Device name
- **GET** `/api/v1/safetymonitor/0/supportedactions` - Supported custom actions (returns [])

### Management API

- **GET** `/management/apiversions` - Supported API versions
- **GET** `/management/v1/description` - Server description
- **GET** `/management/v1/configureddevices` - List of devices

## Device Discovery

The server implements the ASCOM Alpaca UDP Discovery Protocol, allowing ASCOM clients to automatically find the device on the network without manual configuration.

### How Discovery Works

1. **Client broadcasts** a UDP discovery request (`alpacadiscovery1`) to port **32227**
2. **Server responds** with its Alpaca API port number (11111 by default)
3. **Client connects** to the HTTP API using the discovered port

### Discovering the Device

Most ASCOM clients (NINA, SGP, etc.) automatically discover Alpaca devices. If your client supports discovery:

1. Start the SafetyMonitor server
2. Use your client's "Discover Devices" or "Scan Network" feature
3. The "Cloud Detection Safety Monitor" should appear automatically
4. Select it and connect

### Manual Discovery Test

You can test discovery manually using Python:

```python
import socket
import json

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.settimeout(2)

# Send discovery request (broadcast)
sock.sendto(b"alpacadiscovery1", ("<broadcast>", 32227))
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

# Receive response
try:
    data, addr = sock.recvfrom(1024)
    response = json.loads(data)
    print(f"Found Alpaca server at {addr[0]}:{response['AlpacaPort']}")
except socket.timeout:
    print("No response - server may not be running")
```

### Network Requirements

- UDP port **32227** must be accessible for discovery
- HTTP port **11111** (or configured port) must be accessible for API
- When using `network_mode: host` in Docker, both ports are automatically available

## ASCOM Client Integration

### Using Alpyca (Python)

```python
from alpaca.safetymonitor import SafetyMonitor

# Connect to device
monitor = SafetyMonitor('localhost:11111', 0)
monitor.Connected = True

# Check if safe
if monitor.IsSafe:
    print("Conditions are SAFE")
else:
    print("Conditions are UNSAFE")

# Disconnect
monitor.Connected = False
```

### Using ASCOM Remote (Windows)

1. Install ASCOM Remote from https://github.com/ASCOMInitiative/ASCOMRemote/releases
2. Configure the Alpaca device:
   - Host: `localhost` (or server IP)
   - Port: `11111`
   - Device Type: SafetyMonitor
   - Device Number: `0`
3. Use with any ASCOM-compatible application

## Safety Logic

The safety determination is based on the detected cloud condition:

```python
UNSAFE_CONDITIONS = {'Rain', 'Snow', 'Mostly Cloudy', 'Overcast'}

# Returns True if condition is NOT in unsafe set
is_safe = cloud_condition not in UNSAFE_CONDITIONS
```

If no detection has been performed yet, the device returns `False` (unsafe) as a precaution.

## Error Handling

The server follows ASCOM error conventions:

- **ErrorNumber 0**: Success
- **ErrorNumber 0x400 (1024)**: Not Implemented
- **ErrorNumber 0x401 (1025)**: Invalid Value
- **ErrorNumber 0x407 (1031)**: Not Connected
- **ErrorNumber 0x500+**: Server errors

All errors return HTTP 200 with error details in the JSON response (except for 400/500 HTTP errors on malformed requests).

## Logging

The server logs to stdout with INFO level by default:

```
2024-11-30 12:00:00 - INFO - Starting ASCOM Alpaca SafetyMonitor on port 11111
2024-11-30 12:00:05 - INFO - Connected to safety monitor
2024-11-30 12:00:06 - INFO - Cloud detection: Clear (98.5%)
2024-11-30 12:00:36 - INFO - Cloud detection: Wisps of clouds (95.2%)
```

## Deployment

### Docker (Integrated with Existing Container)

The Alpaca SafetyMonitor is integrated into the existing Docker container and runs alongside the MQTT cloud detection service.

**Both services run concurrently in the same container:**
- `detect.py` - Publishes cloud detection to MQTT
- `alpaca_safety_monitor.py` - Serves Alpaca API on port 11111

**Using docker-compose** (existing setup works as-is):

```yaml
services:
    simpleclouddetect:
        container_name: simple-cloud-detect
        network_mode: host  # Port 11111 automatically accessible
        environment:
            - IMAGE_URL=http://allskypi5.lan/current/resized/image.jpg
            - MQTT_BROKER=192.168.1.250
            - MQTT_PORT=1883
            - MQTT_TOPIC=Astro/SimpleCloudDetect
            - DETECT_INTERVAL=60
        volumes:
            - /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5
            - /docker/simpleclouddetect/labels.txt:/app/labels.txt
        restart: unless-stopped
        image: chvvkumar/simpleclouddetect:dev
```

Start the container:
```bash
docker-compose up -d
```

Access the Alpaca API at `http://localhost:11111` (or your host IP).

**Building the image:**

```bash
docker build -t chvvkumar/simpleclouddetect:dev .
```

**What's running:**
- The `start_services.sh` script launches both services
- MQTT detection continues to work unchanged
- Alpaca API is available on port 11111
- Both services share the same model and environment

### Systemd Service

Create `/etc/systemd/system/alpaca-safety-monitor.service`:

```ini
[Unit]
Description=ASCOM Alpaca SafetyMonitor
After=network.target

[Service]
Type=simple
User=observatory
WorkingDirectory=/opt/alpaca-safety-monitor
Environment="IMAGE_URL=http://camera/image.jpg"
Environment="MQTT_BROKER=localhost"
Environment="MQTT_TOPIC=cloud/status"
Environment="DETECT_INTERVAL=60"
Environment="ALPACA_PORT=11111"
ExecStart=/usr/bin/python3 alpaca_safety_monitor.py
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:

```bash
sudo systemctl enable alpaca-safety-monitor
sudo systemctl start alpaca-safety-monitor
```

## Testing

### Manual Testing

Test all endpoints:

```bash
# Get device info
curl "http://localhost:11111/api/v1/safetymonitor/0/name"
curl "http://localhost:11111/api/v1/safetymonitor/0/description"
curl "http://localhost:11111/api/v1/safetymonitor/0/interfaceversion"

# Connect
curl -X PUT "http://localhost:11111/api/v1/safetymonitor/0/connected" -d "Connected=true"

# Check connection
curl "http://localhost:11111/api/v1/safetymonitor/0/connected"

# Get safety status
curl "http://localhost:11111/api/v1/safetymonitor/0/issafe"

# Get device state
curl "http://localhost:11111/api/v1/safetymonitor/0/devicestate"

# Disconnect
curl -X PUT "http://localhost:11111/api/v1/safetymonitor/0/connected" -d "Connected=false"
```

### Using ASCOM Conformance Checker

The ASCOM Conformance Checker can validate full API compliance:
https://ascom-standards.org/Downloads/DeveloperTools.htm

## Troubleshooting

### "Device not connected" error
Ensure you've set `Connected=true` before querying IsSafe.

### No cloud detections
Check that:
- IMAGE_URL is accessible
- Model and labels files exist
- Detection loop is running (check logs)

### Port already in use
Change the port: `export ALPACA_PORT=11112`

### High CPU usage
Increase update interval: `export ALPACA_UPDATE_INTERVAL=60`

## References

- [ASCOM Alpaca API](https://ascom-standards.org/api/)
- [SafetyMonitor Interface](https://ascom-standards.org/newdocs/safetymonitor.html)
- [Alpyca Python Library](https://github.com/ASCOMInitiative/alpyca)
- [ASCOM Standards](https://ascom-standards.org/)

## License

This project follows the same license as the main cloud detection system.

## Contributing

Contributions welcome! Please ensure:
- Full ASCOM API compliance
- Proper error handling
- Thread safety
- Comprehensive logging
