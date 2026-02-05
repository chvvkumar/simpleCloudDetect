# External REST API Documentation

Complete reference for the SimpleCloudDetect External REST API - a flexible interface for dashboards, monitoring systems, and custom integrations.

---

## Table of Contents

- [Overview](#overview)
- [API Base URL](#api-base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [System Information](#system-information)
  - [Status](#status)
  - [Configuration](#configuration)
  - [Connected Clients](#connected-clients)
  - [Safety History](#safety-history)
  - [Latest Image](#latest-image)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Usage Examples](#usage-examples)

---

## Overview

The External REST API provides read-only access to SimpleCloudDetect's internal state, separate from the strict ASCOM Alpaca API. This API is designed for:

- **Dashboards** - Build custom monitoring interfaces
- **Home Automation** - Integrate with non-MQTT systems
- **Monitoring Systems** - Track device health and client connections
- **Custom Scripts** - Automate workflows based on detection data
- **Analytics** - Collect historical safety state transitions

**Key Features:**
- Clean JSON responses with ISO 8601 timestamps
- Thread-safe data access
- No authentication required (designed for trusted networks)
- Independent from ASCOM Alpaca compliance requirements

---

## API Base URL

All endpoints are accessible at:

```
http://<host>:11111/api/ext/v1/
```

Default port: `11111` (configurable via `ALPACA_PORT` environment variable)

---

## Authentication

Currently, the API does not require authentication. It is designed for use within trusted networks. If you need to expose this API externally, consider using a reverse proxy with authentication (e.g., nginx with basic auth, Traefik with middleware).

---

## Endpoints

### System Information

**GET** `/api/ext/v1/system`

Returns system information and uptime statistics.

#### Response

```json
{
  "name": "SimpleCloudDetect",
  "uptime_seconds": 3600.5,
  "uptime_formatted": "1:00:00",
  "server_time": "2026-02-04T15:30:45.123456"
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | System name |
| `uptime_seconds` | number | Seconds since service started |
| `uptime_formatted` | string | Human-readable uptime (HH:MM:SS) |
| `server_time` | string | Current server time (ISO 8601) |

---

### Status

**GET** `/api/ext/v1/status`

Returns current safety status and latest detection results.

#### Response

```json
{
  "is_safe": true,
  "safety_status": "Safe",
  "detection": {
    "class_name": "Clear",
    "confidence_score": 0.9823,
    "Detection Time (Seconds)": 0.234,
    "timestamp": "2026-02-04T15:30:45.123456",
    "Image URL": "http://allsky.local/image.jpg"
  }
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `is_safe` | boolean | Current safety state |
| `safety_status` | string | "Safe" or "Unsafe" |
| `detection.class_name` | string | Detected condition (Clear, Wisps, Mostly Cloudy, etc.) |
| `detection.confidence_score` | number | ML model confidence (0.0 - 1.0) |
| `detection.timestamp` | string | Detection time (ISO 8601) |

---

### Configuration

**GET** `/api/ext/v1/config`

Returns current configuration settings.

#### Response

```json
{
  "device": {
    "name": "SimpleCloudDetect",
    "location": "Home Observatory",
    "id": 0
  },
  "imaging": {
    "url": "http://allsky.local/image.jpg",
    "interval": 30
  },
  "safety": {
    "unsafe_conditions": ["Mostly Cloudy", "Overcast", "Rain", "Snow"],
    "thresholds": {
      "Clear": 0.75,
      "Wisps": 0.70
    },
    "default_threshold": 0.65,
    "debounce_safe_sec": 120,
    "debounce_unsafe_sec": 30
  },
  "system": {
    "timezone": "America/New_York",
    "ntp_server": "pool.ntp.org",
    "update_interval": 30
  }
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `device.name` | string | Device name |
| `device.location` | string | Device location |
| `imaging.url` | string | Image source URL |
| `imaging.interval` | number | Detection interval (seconds) |
| `safety.unsafe_conditions` | array | Conditions that trigger "unsafe" state |
| `safety.thresholds` | object | Per-class confidence thresholds |
| `safety.debounce_safe_sec` | number | Time to wait before switching to "safe" |
| `safety.debounce_unsafe_sec` | number | Time to wait before switching to "unsafe" |

---

### Connected Clients

**GET** `/api/ext/v1/clients`

Returns information about connected ASCOM Alpaca clients.

#### Response

```json
{
  "connected_count": 2,
  "clients": [
    {
      "ip": "192.168.1.100",
      "client_id": 1,
      "connected_at": "2026-02-04T15:00:00.000000",
      "last_seen": "2026-02-04T15:30:00.000000",
      "duration_seconds": 1800.0
    },
    {
      "ip": "192.168.1.101",
      "client_id": 2,
      "connected_at": "2026-02-04T15:15:00.000000",
      "last_seen": "2026-02-04T15:30:00.000000",
      "duration_seconds": 900.0
    }
  ]
}
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `connected_count` | number | Number of active clients |
| `clients[].ip` | string | Client IP address |
| `clients[].client_id` | number | ASCOM client ID |
| `clients[].connected_at` | string | Connection start time (ISO 8601) |
| `clients[].last_seen` | string | Last heartbeat time (ISO 8601) |
| `clients[].duration_seconds` | number | Connection duration in seconds |

---

### Safety History

**GET** `/api/ext/v1/history`

Returns safety state transition history (newest first).

#### Response

```json
[
  {
    "timestamp": "2026-02-04T15:30:00.000000",
    "state": "Safe",
    "condition": "Clear",
    "confidence": 0.98
  },
  {
    "timestamp": "2026-02-04T15:00:00.000000",
    "state": "Unsafe",
    "condition": "Overcast",
    "confidence": 0.92
  }
]
```

#### Fields

| Field | Type | Description |
|-------|------|-------------|
| `timestamp` | string | Transition time (ISO 8601) |
| `state` | string | "Safe" or "Unsafe" |
| `condition` | string | Detected condition at transition |
| `confidence` | number | ML model confidence (0.0 - 1.0) |

---

### Latest Image

**GET** `/api/ext/v1/image`

Returns the latest detection image as raw JPEG data.

#### Response

- **Content-Type**: `image/jpeg`
- **Body**: Raw JPEG image bytes

#### Error Response

If no image is available:

```json
{
  "error": "No image available"
}
```

**Status Code**: 404

---

## Response Formats

### Success Response

All successful responses return JSON with HTTP status `200 OK`.

### Error Response

When the system is not initialized:

```json
{
  "error": "System not initialized"
}
```

**Status Code**: 503 Service Unavailable

---

## Error Handling

| Status Code | Description |
|-------------|-------------|
| `200` | Success |
| `404` | Resource not found (e.g., no image available) |
| `503` | Service unavailable (system not initialized) |

---

## Usage Examples

### cURL

**Check current status:**
```bash
curl http://localhost:11111/api/ext/v1/status
```

**Get system uptime:**
```bash
curl http://localhost:11111/api/ext/v1/system
```

**Download latest image:**
```bash
curl http://localhost:11111/api/ext/v1/image -o latest.jpg
```

### Python

```python
import requests

# Get current status
response = requests.get("http://localhost:11111/api/ext/v1/status")
data = response.json()

if data["is_safe"]:
    print(f"Sky is safe: {data['detection']['class_name']}")
else:
    print(f"Sky is unsafe: {data['detection']['class_name']}")

# Get safety history
history = requests.get("http://localhost:11111/api/ext/v1/history").json()
print(f"Last {len(history)} state transitions:")
for entry in history[:5]:  # Show last 5
    print(f"  {entry['timestamp']}: {entry['state']} ({entry['condition']})")
```

### JavaScript (Node.js)

```javascript
const fetch = require('node-fetch');

async function checkStatus() {
  const response = await fetch('http://localhost:11111/api/ext/v1/status');
  const data = await response.json();
  
  console.log(`Safety: ${data.safety_status}`);
  console.log(`Condition: ${data.detection.class_name}`);
  console.log(`Confidence: ${(data.detection.confidence_score * 100).toFixed(1)}%`);
}

checkStatus();
```

### Home Assistant REST Sensor

Add to `configuration.yaml`:

```yaml
sensor:
  - platform: rest
    name: "Sky Safety Status"
    resource: "http://192.168.1.100:11111/api/ext/v1/status"
    value_template: "{{ value_json.safety_status }}"
    json_attributes:
      - is_safe
      - detection
    scan_interval: 30

  - platform: rest
    name: "Connected Clients"
    resource: "http://192.168.1.100:11111/api/ext/v1/clients"
    value_template: "{{ value_json.connected_count }}"
    json_attributes:
      - clients
    scan_interval: 60
```

---

## Integration Tips

### Polling Recommendations

- **Status endpoint**: Poll at your detection interval (typically 30-60 seconds)
- **System endpoint**: Poll every 5-10 minutes for monitoring
- **Clients endpoint**: Poll every 1-2 minutes to track connections
- **History endpoint**: Poll only when needed (e.g., on-demand or hourly)
- **Image endpoint**: Request only when needed; can be bandwidth-intensive

### Network Security

Since this API has no authentication:

1. **Use a firewall** to restrict access to trusted networks
2. **Deploy a reverse proxy** (nginx, Traefik) with authentication if exposing externally
3. **Use HTTPS** if transmitting over untrusted networks
4. **Consider VPN** for remote access instead of direct exposure

### Performance Considerations

- All endpoints are thread-safe and designed for concurrent access
- Image endpoint returns the last processed image (no on-demand processing)
- History size is limited to prevent memory issues (typically last 100 transitions)
- Consider caching responses in your client to reduce load

---

## See Also

- [ASCOM Alpaca SafetyMonitor API](ALPACA_README.md) - Standard ASCOM API documentation
- [Main README](../readme.md) - Installation and general usage
- [Architecture](../architecture.md) - System design and internals
