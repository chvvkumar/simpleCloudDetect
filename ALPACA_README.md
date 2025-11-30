# ASCOM Alpaca SafetyMonitor Implementation

## Overview

The Alpaca SafetyMonitor implementation extends the SimpleCloudDetect project to provide an **ASCOM Alpaca-compliant SafetyMonitor** device. This allows astronomy software (like N.I.N.A., SGP, TheSkyX, etc.) to automatically monitor sky conditions and protect expensive equipment by pausing operations when conditions become unsafe.

## Features

- **Full ASCOM Alpaca Compliance**: Implements the ISafetyMonitor interface v1
- **ML-Based Cloud Detection**: Uses machine learning to classify cloud conditions
- **Automatic UDP Discovery**: ASCOM clients can automatically discover the device on your network
- **Web-Based Setup Interface**: Configure device settings through a web browser
- **Configurable Safety Criteria**: Define which cloud conditions are safe/unsafe
- **Real-time Monitoring**: Continuous background monitoring with configurable update intervals
- **Persistent Configuration**: Settings saved to file and preserved across restarts
- **Docker Support**: Runs alongside the MQTT cloud detection service

## Architecture

The implementation consists of three main components:

1. **Cloud Detection Engine** (`detect.py`)
   - Analyzes AllSky camera images using a trained ML model
   - Classifies conditions: Clear, Wisps of clouds, Mostly Cloudy, Overcast, Rain, Snow
   - Provides confidence scores for each prediction

2. **Alpaca SafetyMonitor Server** (`alpaca_safety_monitor.py`)
   - Flask-based HTTP API server implementing ASCOM Alpaca protocol
   - Background thread for continuous cloud detection
   - UDP discovery service for automatic device detection
   - Web-based configuration interface

3. **Service Manager** (`start_services.sh`)
   - Runs both MQTT and Alpaca services simultaneously
   - Ensures model conversion on startup
   - Uses Gunicorn for production-grade HTTP serving

## Docker Deployment

### Environment Variables

#### Cloud Detection Settings (Shared)
- `IMAGE_URL`: URL or file path to AllSky camera image (required)
- `MQTT_BROKER`: MQTT broker address (required for MQTT service)
- `MQTT_PORT`: MQTT broker port (default: 1883)
- `MQTT_TOPIC`: MQTT topic for publishing (default: Astro/SimpleCloudDetect)
- `MQTT_USERNAME`: MQTT authentication username (optional)
- `MQTT_PASSWORD`: MQTT authentication password (optional)
- `DETECT_INTERVAL`: Detection interval in seconds (default: 60)

#### Alpaca-Specific Settings
- `ALPACA_PORT`: HTTP API port (default: 11111)
- `ALPACA_DEVICE_NUMBER`: Device number for multi-device setups (default: 0)
- `ALPACA_UPDATE_INTERVAL`: Update interval in seconds (default: 30)

### Docker Run Example

```bash
docker run -d --name simple-cloud-detect \
  --network=host \
  -e IMAGE_URL="http://allskypi.lan/current/image.jpg" \
  -e MQTT_BROKER="192.168.1.100" \
  -e MQTT_PORT="1883" \
  -e MQTT_TOPIC="Astro/SimpleCloudDetect" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  -e DETECT_INTERVAL="60" \
  -e ALPACA_PORT="11111" \
  -e ALPACA_UPDATE_INTERVAL="30" \
  -v /path/to/keras_model.h5:/app/keras_model.h5 \
  -v /path/to/labels.txt:/app/labels.txt \
  chvvkumar/simpleclouddetect:latest
```

### Docker Compose Example

```yaml
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    network_mode: host
    environment:
      - IMAGE_URL=http://allskypi.lan/current/image.jpg
      - MQTT_BROKER=192.168.1.100
      - MQTT_PORT=1883
      - MQTT_TOPIC=Astro/SimpleCloudDetect
      - MQTT_USERNAME=your_username
      - MQTT_PASSWORD=your_password
      - DETECT_INTERVAL=60
      - ALPACA_PORT=11111
      - ALPACA_UPDATE_INTERVAL=30
    volumes:
      - /path/to/keras_model.h5:/app/keras_model.h5
      - /path/to/labels.txt:/app/labels.txt
    restart: unless-stopped
    image: chvvkumar/simpleclouddetect:latest
```

**Note**: `network_mode: host` is recommended for UDP discovery to work properly.

### My own compose file example that I run

```yaml
services:
    simpleclouddetect:
        container_name: simple-cloud-detect
        network_mode: host
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
        image: chvvkumar/simpleclouddetect:dev #latest
```

## Configuration

### Initial Setup

After starting the container, access the web-based setup interface:

```
http://<your-server-ip>:11111/setup/v1/safetymonitor/0/setup
```

### Configurable Parameters

1. **Device Name**: Friendly name displayed in ASCOM clients
2. **Location**: Physical location of the device (e.g., "Backyard Observatory")
3. **Unsafe Conditions**: Select which cloud conditions should trigger unsafe status
   - Clear ❌ (typically safe)
   - Wisps of clouds ❌ (typically safe)
   - Mostly Cloudy ✅ (typically unsafe)
   - Overcast ✅ (typically unsafe)
   - Rain ✅ (unsafe)
   - Snow ✅ (unsafe)

### Configuration Persistence

Settings are saved to `alpaca_config.json` in the container. To persist across container recreations:

```bash
-v /path/to/config:/app/alpaca_config.json
```

## ASCOM Client Setup

### Automatic Discovery

Most ASCOM clients support automatic discovery:

1. Open your ASCOM client (N.I.N.A., SGP, etc.)
2. Add a new SafetyMonitor device
3. Choose "ASCOM Alpaca Discovery"
4. Select "SimpleCloudDetect" from discovered devices
5. Test connection

### Manual Configuration

If automatic discovery doesn't work:

1. Add new SafetyMonitor device
2. Choose "ASCOM Alpaca"
3. Enter manually:
   - **Host**: `<your-server-ip>`
   - **Port**: `11111` (or your ALPACA_PORT)
   - **Device Number**: `0` (or your ALPACA_DEVICE_NUMBER)
   - **Device Type**: SafetyMonitor

## API Endpoints

### ASCOM Alpaca Standard Endpoints

#### Safety Status
```http
GET /api/v1/safetymonitor/0/issafe
```
Returns: `true` if conditions are safe, `false` if unsafe

#### Connection Control
```http
GET /api/v1/safetymonitor/0/connected
PUT /api/v1/safetymonitor/0/connected?Connected=true
```

#### Device Information
```http
GET /api/v1/safetymonitor/0/name
GET /api/v1/safetymonitor/0/description
GET /api/v1/safetymonitor/0/driverinfo
GET /api/v1/safetymonitor/0/driverversion
GET /api/v1/safetymonitor/0/interfaceversion
```

#### Device State (Platform 7)
```http
GET /api/v1/safetymonitor/0/devicestate
```
Returns detailed state including:
- IsSafe status
- Current cloud condition
- Confidence score
- Last update timestamp

### Management Endpoints

```http
GET /management/apiversions
GET /management/v1/description
GET /management/v1/configureddevices
```

### Setup Interface

```http
GET  /setup/v1/safetymonitor/0/setup  # View configuration page
POST /setup/v1/safetymonitor/0/setup  # Save configuration
```

## How It Works

### Detection Loop

1. When connected, the SafetyMonitor starts a background thread
2. Thread continuously fetches images from `IMAGE_URL`
3. Images are analyzed using the ML model
4. Results are stored with timestamp and confidence score
5. Loop repeats every `ALPACA_UPDATE_INTERVAL` seconds

### Safety Determination

```python
# Pseudo-code
latest_condition = get_latest_detection()
is_safe = latest_condition NOT IN unsafe_conditions

# Example:
# Detected: "Mostly Cloudy" (87% confidence)
# Unsafe conditions: [Rain, Snow, Mostly Cloudy, Overcast]
# Result: is_safe = False
```

### Client Polling

ASCOM clients typically poll the `issafe` endpoint every 10-60 seconds to check current conditions.

## Safety Considerations

### Default Unsafe Conditions

By default, these conditions are considered **UNSAFE**:
- Rain
- Snow
- Mostly Cloudy
- Overcast

### Disconnected Behavior

**IMPORTANT**: When disconnected or unable to fetch images, the SafetyMonitor always returns `false` (unsafe) to protect equipment.

### Update Frequency

- **ALPACA_UPDATE_INTERVAL**: How often cloud detection runs (default: 30s)
- **Client polling**: Controlled by ASCOM client software (typically 30-60s)

Consider your needs:
- **Faster updates** (10-15s): Better response time, more CPU/network usage
- **Slower updates** (60-120s): Less overhead, slower response to changing conditions

## Troubleshooting

### Discovery Not Working

1. Ensure `network_mode: host` in Docker
2. Check firewall allows UDP 32227
3. Verify client and server on same network/VLAN
4. Try manual configuration as fallback

### Connection Refused

1. Check ALPACA_PORT is correct
2. Verify container is running: `docker ps`
3. Check container logs: `docker logs simple-cloud-detect`
4. Test API directly: `curl http://localhost:11111/management/v1/description`

### Always Showing Unsafe

1. Check IMAGE_URL is accessible from container
2. Verify model files are properly mounted
3. Review safety configuration in setup page
4. Check logs for detection errors

### No Updates

1. Verify ALPACA_UPDATE_INTERVAL is set appropriately
2. Check if detection loop is running (logs should show "Cloud detection: ...")
3. Ensure image at IMAGE_URL is updating

## Integration Examples

### N.I.N.A. (Nighttime Imaging 'N' Astronomy)

1. Options → Equipment → Safety Monitor
2. Choose "ASCOM SafetyMonitor"
3. Select "SimpleCloudDetect" from chooser
4. Enable "Use Safety Monitor"
5. Configure unsafe actions (Park, Warm camera, etc.)

### Sequence Generator Pro

1. Tools → Device Setup → Safety Monitor
2. Select "ASCOM SafetyMonitor"
3. Choose "SimpleCloudDetect"
4. Enable safety monitoring in sequence

### TheSkyX

1. Tools → Device Setup → Safety Monitor
2. Add new SafetyMonitor
3. Choose Alpaca device type
4. Configure connection details

## Performance Notes

- **Memory**: ~200-500MB depending on model size
- **CPU**: Minimal (detection runs every 30-60s)
- **Network**: Minimal (single image fetch per interval)
- **Startup time**: 5-10 seconds

## Logging

Logs include:
- Connection/disconnection events
- Cloud detection results with confidence scores
- Configuration changes
- API requests (client/transaction IDs)
- Errors and warnings

View logs:
```bash
docker logs -f simple-cloud-detect
```

## Advanced Configuration

### Custom Model

Mount your own trained model:
```bash
-v /path/to/custom_model.h5:/app/keras_model.h5
-v /path/to/custom_labels.txt:/app/labels.txt
```

### Multiple Instances

Run multiple instances for different cameras:
```bash
# Instance 1
docker run -d --name cloud-detect-east \
  -e ALPACA_PORT=11111 \
  -e ALPACA_DEVICE_NUMBER=0 \
  -e IMAGE_URL="http://east-camera.lan/image.jpg" \
  ...

# Instance 2
docker run -d --name cloud-detect-west \
  -e ALPACA_PORT=11112 \
  -e ALPACA_DEVICE_NUMBER=0 \
  -e IMAGE_URL="http://west-camera.lan/image.jpg" \
  ...
```

## Contributing

Contributions welcome! Areas for improvement:
- Additional cloud condition classifications
- Machine learning model improvements
- Enhanced configuration options
- Support for multiple cameras per instance


## Credits

- **Author**: chvvkumar
- **Based on**: SimpleCloudDetect cloud detection
- **Protocol**: ASCOM Alpaca Standard
- **Repository**: https://github.com/chvvkumar/simpleCloudDetect

## Support

For issues, questions, or contributions:
- GitHub Issues: https://github.com/chvvkumar/simpleCloudDetect/issues
- Discussions: https://github.com/chvvkumar/simpleCloudDetect/discussions
