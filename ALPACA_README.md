# ASCOM Alpaca SafetyMonitor Guide

Complete documentation for the ASCOM Alpaca SafetyMonitor implementation in SimpleCloudDetect.

---

## Table of Contents

- [Overview](#overview)
- [What is ASCOM Alpaca?](#what-is-ascom-alpaca)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
  - [Environment Variables](#environment-variables)
  - [Web Setup Interface](#web-setup-interface)
- [Client Configuration](#client-configuration)
  - [N.I.N.A. Setup](#nina-setup)
  - [Sequence Generator Pro](#sequence-generator-pro)
  - [TheSkyX](#theskyx)
  - [Generic ASCOM Client](#generic-ascom-client)
- [Safety Logic](#safety-logic)
- [API Reference](#api-reference)
- [Troubleshooting](#troubleshooting)

---

## Overview

The ASCOM Alpaca SafetyMonitor service provides real-time sky condition monitoring for astronomy automation software. It translates ML-based cloud detection results into standard ASCOM SafetyMonitor states, allowing your imaging software to automatically pause or resume operations based on sky conditions.

**Key Features:**
- Standard ASCOM Alpaca REST API implementation
- Configurable unsafe conditions (which sky states trigger "unsafe")
- Web-based configuration interface
- Automatic integration with N.I.N.A., SGP, TheSkyX, and other ASCOM clients
- Independent operation alongside MQTT publishing

---

## What is ASCOM Alpaca?

ASCOM Alpaca is a modern, platform-independent protocol for astronomy device control. It uses REST APIs over HTTP, making it compatible with:

- **Windows, Linux, and macOS** astronomy software
- **Network-based** device access (no USB required)
- **Multiple simultaneous connections** from different applications
- **Cross-platform compatibility** without COM/DCOM dependencies

The SafetyMonitor device type allows automation software to monitor environmental conditions and automatically take protective actions when conditions become unsafe.

---

## Quick Start

### 1. Start the Container

The Alpaca service runs automatically when you start the SimpleCloudDetect container:

```shell
docker run -d --name simple-cloud-detect --network=host \
  -e IMAGE_URL="http://your-allsky-camera/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DETECT_INTERVAL="60" \
  -e DEVICE_ID="clouddetect_001" \
  -e ALPACA_PORT="11111" \
  chvvkumar/simpleclouddetect:latest
```

### 2. Access Setup Page

Open your browser and navigate to:
```
http://<your-server-ip>:11111/setup/v1/safetymonitor/0/setup
```

Example: `http://192.168.1.100:11111/setup/v1/safetymonitor/0/setup`

### 3. Configure Unsafe Conditions

Select which sky conditions should be considered "unsafe" (typically cloudy conditions that should pause imaging).

### 4. Add to Your Software

In your astronomy software (N.I.N.A., SGP, etc.), add a new ASCOM Alpaca SafetyMonitor device:
- **Host**: Your server's IP address
- **Port**: `11111` (or your custom `ALPACA_PORT`)
- **Device Number**: `0` (or your custom `ALPACA_DEVICE_NUMBER`)

---

## Configuration

### Environment Variables

Configure the Alpaca service using these Docker environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_PORT` | `11111` | HTTP port for the Alpaca API server |
| `ALPACA_DEVICE_NUMBER` | `0` | Device number (0-99, use different numbers for multiple devices) |
| `ALPACA_UPDATE_INTERVAL` | `30` | How often to update sky conditions (seconds) |

**Example with custom settings:**
```yaml
services:
  simpleclouddetect:
    image: chvvkumar/simpleclouddetect:latest
    network_mode: host
    environment:
      - IMAGE_URL=http://allskypi.lan/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_DISCOVERY_MODE=homeassistant
      - DETECT_INTERVAL=60
      - DEVICE_ID=clouddetect_001
      - ALPACA_PORT=11111
      - ALPACA_DEVICE_NUMBER=0
      - ALPACA_UPDATE_INTERVAL=30
```

### Web Setup Interface

The web interface provides configuration and status monitoring:

**Setup Page**: `http://<server-ip>:<port>/setup/v1/safetymonitor/<device>/setup`

**Features:**
- **Device Information** - View device name, description, and connection details
- **Unsafe Conditions** - Select which sky conditions trigger unsafe status
- **Current Status** - Real-time display of current sky condition and safety state
- **Connection Test** - Verify device is responding correctly

**Default Unsafe Conditions:**
- Mostly Cloudy
- Overcast
- Rain
- Snow

> **Note:** Changes take effect immediately without requiring a restart.

---

## Client Configuration

### N.I.N.A. Setup

**Nighttime Imaging 'N' Astronomy (N.I.N.A.)** supports ASCOM Alpaca devices natively.

1. Open N.I.N.A.
2. Go to **Equipment** → **Safety Monitor**
3. Click **Choose**
4. Select **ASCOM Alpaca Discovery** or **Manual Configuration**

**Automatic Discovery:**
- Click **Discover Devices**
- Select "SimpleCloudDetect SafetyMonitor" from the list
- Click **Connect**

**Manual Configuration:**
- Select **ASCOM.Alpaca.SafetyMonitor**
- Configure:
  - **Host**: `192.168.1.100` (your server IP)
  - **Port**: `11111`
  - **Device Number**: `0`
- Click **Connect**

**Using in Sequences:**
- Add **Wait for Safe Conditions** instruction to your sequence
- N.I.N.A. will automatically pause when sky becomes unsafe
- Imaging resumes when conditions become safe again

### Sequence Generator Pro

**Sequence Generator Pro (SGP)** requires ASCOM Platform 6.5 or later with Alpaca support.

1. Open Sequence Generator Pro
2. Go to **Tools** → **Options** → **ASCOM**
3. Click **Manage Alpaca Devices**
4. Add new device:
   - **IP Address**: Your server IP
   - **Port**: `11111`
   - **Device Type**: SafetyMonitor
   - **Device Number**: `0`
5. In the main window, select the SafetyMonitor from the device dropdown
6. Enable **Monitor Safety Device** in your sequence settings

**Sequence Integration:**
- SGP will automatically pause the sequence when unsafe
- Optionally configure warm-up behavior when resuming
- Set notification preferences for safety state changes

### TheSkyX

**TheSkyX** supports ASCOM Alpaca devices through its device manager.

1. Open TheSkyX
2. Go to **Tools** → **Device Manager**
3. Add new device:
   - **Type**: Safety Monitor
   - **Protocol**: ASCOM Alpaca
   - **Host**: Your server IP
   - **Port**: `11111`
   - **Device**: `0`
4. Click **Connect**

**Automation:**
- Use JavaScript or visual scripting to check safety status
- Implement custom logic to pause/resume imaging
- Integrate with other TheSkyX automation features

### Generic ASCOM Client

Any ASCOM Alpaca-compatible client can connect using these details:

**Connection URL Format:**
```
http://<server-ip>:<port>/api/v1/safetymonitor/<device>/<method>
```

**Example:**
```
http://192.168.1.100:11111/api/v1/safetymonitor/0/issafe
```

**ASCOM Discovery:**
- Most modern ASCOM clients support automatic Alpaca device discovery
- Enable discovery on UDP port 32227
- Devices appear automatically in client software

---

## Safety Logic

### Sky Condition Mapping

The SafetyMonitor translates cloud detection results into safety states:

| Sky Condition | Default State | Typical Recommendation |
|--------------|---------------|------------------------|
| **Clear** | Safe | Continue imaging |
| **Wisps** | Safe | Continue imaging (monitor) |
| **Mostly Cloudy** | Unsafe | Pause and wait |
| **Overcast** | Unsafe | Pause and close |
| **Rain** | Unsafe | Pause and close immediately |
| **Snow** | Unsafe | Pause and close immediately |

> **Note:** You can customize which conditions are considered unsafe via the web setup interface.

### Safety State Logic

**Safe Conditions:**
- `IsSafe` returns `true`
- Imaging software continues normal operations
- No alerts or warnings generated

**Unsafe Conditions:**
- `IsSafe` returns `false`
- Imaging software should:
  - Pause current exposures
  - Park the mount
  - Close dust covers or roof
  - Optionally warm the camera
- Alerts/notifications triggered based on client configuration

**Confidence Threshold:**
- Detection confidence of >70% is considered reliable
- Lower confidence maintains previous state
- Prevents rapid state changes from marginal detections

---

## API Reference

### Standard ASCOM Alpaca Endpoints

**Base URL:** `http://<server-ip>:<port>/api/v1/safetymonitor/<device>`

#### Device Information

**GET** `/connected`
- Returns connection status
- Response: `{"Value": true, "ErrorNumber": 0, "ErrorMessage": ""}`

**GET** `/name`
- Returns device name
- Response: `{"Value": "SimpleCloudDetect SafetyMonitor", ...}`

**GET** `/description`
- Returns device description
- Response: `{"Value": "ML-based cloud detection safety monitor", ...}`

**GET** `/driverinfo`
- Returns driver information
- Response: `{"Value": "SimpleCloudDetect Alpaca Driver v1.0", ...}`

**GET** `/driverversion`
- Returns driver version
- Response: `{"Value": "1.0", ...}`

#### Safety Monitoring

**GET** `/issafe`
- Returns current safety state
- Response: `{"Value": true, "ErrorNumber": 0, "ErrorMessage": ""}`
- `true` = safe to operate, `false` = unsafe conditions

#### Management Endpoints

**PUT** `/connected`
- Connect/disconnect from device
- Body: `{"Connected": true}`
- Response: `{"ErrorNumber": 0, "ErrorMessage": ""}`

**PUT** `/action`
- Execute device-specific commands
- Body: `{"Action": "command", "Parameters": "params"}`

### Custom Setup Endpoints

**GET** `/setup/v1/safetymonitor/<device>/setup`
- Web interface for device configuration
- Returns HTML configuration page

**POST** `/setup/v1/safetymonitor/<device>/setup`
- Save configuration changes
- Body: Form data with unsafe condition selections

### Discovery Protocol

**UDP Broadcast** on port `32227`
- Responds to Alpaca discovery requests
- Provides device type, host, port, and unique ID
- Enables automatic device detection in client software

---

## Troubleshooting

### Connection Issues

**Problem:** Client cannot connect to SafetyMonitor

**Solutions:**
1. **Verify container is running:**
   ```shell
   docker ps | grep simple-cloud-detect
   ```

2. **Check port accessibility:**
   ```shell
   curl http://localhost:11111/api/v1/safetymonitor/0/issafe
   ```

3. **Verify firewall rules:**
   - Ensure port 11111 (or your custom port) is open
   - Check both host firewall and Docker network settings

4. **Test from client machine:**
   ```shell
   curl http://<server-ip>:11111/api/v1/safetymonitor/0/issafe
   ```

5. **Check Docker network mode:**
   - Use `--network=host` for simplest configuration
   - Or properly map ports with `-p 11111:11111`

### Device Not Appearing in Client

**Problem:** SafetyMonitor not found during discovery

**Solutions:**
1. **Manual configuration:**
   - Use manual device entry instead of auto-discovery
   - Enter IP address, port, and device number directly

2. **Check UDP port 32227:**
   - Discovery uses UDP broadcast
   - Ensure port 32227 is not blocked
   - Some networks block UDP broadcast traffic

3. **Verify ASCOM Platform version:**
   - Ensure ASCOM Platform 6.5 or later is installed
   - Update ASCOM Platform if necessary

### Incorrect Safety Status

**Problem:** Device reports wrong safe/unsafe state

**Solutions:**
1. **Check current detection:**
   - View web setup page for current sky condition
   - Verify detection matches actual conditions

2. **Review unsafe condition settings:**
   - Access setup page
   - Ensure correct conditions are marked as unsafe

3. **Verify model accuracy:**
   - Consider training a custom model for your camera
   - See main README for model training instructions

4. **Check confidence threshold:**
   - Low confidence detections may cause issues
   - Review detection logs for confidence scores

### API Errors

**Problem:** API returns error responses

**Common Error Codes:**
- `400` - Invalid request parameter
- `404` - Device or method not found
- `500` - Internal server error

**Solutions:**
1. **Check API endpoint format:**
   ```
   http://<ip>:<port>/api/v1/safetymonitor/<device>/<method>
   ```

2. **Verify device number:**
   - Default is `0`
   - Must match `ALPACA_DEVICE_NUMBER` environment variable

3. **Review container logs:**
   ```shell
   docker logs simple-cloud-detect
   ```

4. **Check for concurrent requests:**
   - Alpaca supports multiple connections
   - But excessive requests may cause timeouts

### Performance Issues

**Problem:** Slow response or timeouts

**Solutions:**
1. **Adjust update interval:**
   - Increase `ALPACA_UPDATE_INTERVAL` for less frequent updates
   - Reduces CPU and network load

2. **Check detection interval:**
   - Ensure `DETECT_INTERVAL` is appropriate
   - Very short intervals increase system load

3. **Monitor system resources:**
   ```shell
   docker stats simple-cloud-detect
   ```

4. **Review network latency:**
   - Test ping times to server
   - Check for network congestion

---

## Additional Resources

**ASCOM Standards:**
- ASCOM Initiative: https://ascom-standards.org/
- Alpaca API Specification: https://ascom-standards.org/api/
- SafetyMonitor Interface: https://ascom-standards.org/Help/Developer/html/T_ASCOM_DeviceInterface_ISafetyMonitor.htm

**SimpleCloudDetect Documentation:**
- Main README: [readme.md](readme.md)
- GitHub Repository: https://github.com/chvvkumar/simpleCloudDetect

**Support:**
- Report issues on GitHub
- Community discussions in Issues section
- Pull requests welcome for improvements

---

## Version History

- **v1.0** (2025-01-30) - Initial Alpaca SafetyMonitor implementation
  - Standard ASCOM Alpaca REST API
  - Web-based configuration interface
  - Configurable unsafe conditions
  - Automatic discovery support
