# SimpleCloudDetect

A Machine Learning-based cloud detection system for AllSky cameras with MQTT and ASCOM Alpaca SafetyMonitor integration.

[![main](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/main.yml/badge.svg)](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/main.yml) [![Docker Image Size (latest)](https://img.shields.io/docker/image-size/chvvkumar/simpleclouddetect/latest?style=flat&logo=docker&logoSize=auto)](https://hub.docker.com/r/chvvkumar/simpleclouddetect) [![Docker Pulls](https://img.shields.io/docker/pulls/chvvkumar/simpleclouddetect?style=flat&logo=docker&label=Pulls)](https://hub.docker.com/r/chvvkumar/simpleclouddetect)

[![dev](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/dev.yml/badge.svg)](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/dev.yml) [![Docker Image Size (dev)](https://img.shields.io/docker/image-size/chvvkumar/simpleclouddetect/dev?style=flat&logo=docker&logoSize=auto)](https://hub.docker.com/r/chvvkumar/simpleclouddetect/tags)

[![optimization](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/optimization.yml/badge.svg)](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/optimization.yml) [![Docker Image Size (optimization)](https://img.shields.io/docker/image-size/chvvkumar/simpleclouddetect/optimization?style=flat&logo=docker&logoSize=auto)](https://hub.docker.com/r/chvvkumar/simpleclouddetect/tags)

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Screenshots](#screenshots)
- [Docker Installation (Recommended)](#docker-installation-recommended)
  - [Environment Variables](#environment-variables)
  - [Docker Run Examples](#docker-run-examples)
  - [Docker Compose Examples](#docker-compose-examples)
- [Home Assistant Integration](#home-assistant-integration)
- [ASCOM Alpaca SafetyMonitor](#ascom-alpaca-safetymonitor)
- [Manual Installation](#manual-installation-non-docker)
- [Training Your Own Model](#training-your-own-model)
- [Recent Changes](#recent-changes)

---

## Features

- **ML Cloud Classification** - Detects Clear, Wisps, Mostly Cloudy, Overcast, Rain, and Snow conditions
- **Home Assistant Integration** - MQTT Discovery for automatic setup or legacy manual configuration
- **ASCOM Alpaca SafetyMonitor** - Compatible with N.I.N.A., SGP, TheSkyX, and other astronomy software
- **Docker Support** - Easy deployment with both services running simultaneously
- **Flexible Image Sources** - Supports URL-based and local file images
- **Custom Models** - Bring your own trained model and labels
- **Confidence Scores** - Includes detection confidence and timing metrics

---

## Quick Start

**Docker (Recommended):**

```shell
docker run -d --name simple-cloud-detect --network=host \
  -e IMAGE_URL="http://your-allsky-camera/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DEVICE_ID="clouddetect_001" \
  chvvkumar/simpleclouddetect:latest
```

That's it! Your device will automatically appear in Home Assistant under **Settings → Devices & Services → MQTT**.

---

## Screenshots

| Condition | Example |
|-----------|---------|
| **Clear Skies** | ![Clear](/images/HA2.png) |
| **Majority Clouds** | ![Mostly Cloudy](/images/Mostly%20Cloudy.png) |
| **Wisps of Clouds** | ![Wisps](/images/wisps.png) |
| **Overcast** | ![Overcast](/images/Overcast.png) |

---

## Docker Installation (Recommended)

### Pull the Image

```shell
docker pull chvvkumar/simpleclouddetect:latest
```

### Environment Variables

#### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `IMAGE_URL` | URL or file path to AllSky camera image | `http://allskypi.lan/image.jpg` |
| `MQTT_BROKER` | MQTT broker address | `192.168.1.250` |

#### Cloud Detection Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_USERNAME` | - | MQTT authentication username (optional) |
| `MQTT_PASSWORD` | - | MQTT authentication password (optional) |
| `DETECT_INTERVAL` | `60` | Detection interval in seconds |

#### MQTT Publishing Modes

| Variable | Default | Description |
|----------|---------|-------------|
| `MQTT_DISCOVERY_MODE` | `legacy` | Mode: `legacy` or `homeassistant` |

**Legacy Mode** (manual YAML configuration):
- `MQTT_TOPIC` - Topic for publishing (e.g., `Astro/SimpleCloudDetect`)

**Home Assistant Discovery Mode** (automatic setup):
- `DEVICE_ID` - Unique device identifier (e.g., `clouddetect_001`)
- `DEVICE_NAME` - Custom device name (default: `Cloud Detector`)
- `MQTT_DISCOVERY_PREFIX` - HA discovery prefix (default: `homeassistant`)

#### ASCOM Alpaca Settings (Optional)

| Variable | Default | Description |
|----------|---------|-------------|
| `ALPACA_PORT` | `11111` | HTTP API port |
| `ALPACA_DEVICE_NUMBER` | `0` | Device number |
| `ALPACA_UPDATE_INTERVAL` | `30` | Update interval in seconds |

> **Note:** For detailed Alpaca configuration, see **[ALPACA_README.md](ALPACA_README.md)**

### Docker Run Examples

#### Home Assistant Discovery Mode (Recommended)

**With URL-based image:**
```shell
docker run -d --name simple-cloud-detect --network=host \
  -e IMAGE_URL="http://allskypi5.lan/current/resized/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_PORT="1883" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DEVICE_ID="clouddetect_001" \
  -e DEVICE_NAME="AllSky Cloud Detector" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  -e DETECT_INTERVAL="60" \
  chvvkumar/simpleclouddetect:latest
```

**With local file:**
```shell
docker run -d --name simple-cloud-detect --network=host \
  -v $HOME/path/to/image.jpg:/tmp/image.jpg \
  -e IMAGE_URL="file:///tmp/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DEVICE_ID="clouddetect_001" \
  chvvkumar/simpleclouddetect:latest
```

#### Legacy Mode

**With URL-based image:**
```shell
docker run -d --name simple-cloud-detect --network=host \
  -e IMAGE_URL="http://allskypi5.lan/current/resized/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_TOPIC="Astro/SimpleCloudDetect" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  chvvkumar/simpleclouddetect:latest
```

#### Custom Model Support

To use your own trained model and labels:

```shell
docker run -d --name simple-cloud-detect --network=host \
  -v /path/to/your/keras_model.h5:/app/keras_model.h5 \
  -v /path/to/your/labels.txt:/app/labels.txt \
  -e IMAGE_URL="http://allskypi5.lan/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DEVICE_ID="clouddetect_001" \
  chvvkumar/simpleclouddetect:latest
```

### Docker Compose Examples

#### Home Assistant Discovery Mode

```yaml
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    image: chvvkumar/simpleclouddetect:latest
    network_mode: host
    restart: unless-stopped
    environment:
      - IMAGE_URL=http://allskypi5.lan/current/resized/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_PORT=1883
      - MQTT_DISCOVERY_MODE=homeassistant
      - DEVICE_ID=clouddetect_001
      - DEVICE_NAME=AllSky Cloud Detector
      - MQTT_USERNAME=your_username
      - MQTT_PASSWORD=your_password
      - DETECT_INTERVAL=60
      - ALPACA_PORT=11111
      - ALPACA_UPDATE_INTERVAL=30
```

#### Legacy Mode

```yaml
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    image: chvvkumar/simpleclouddetect:latest
    network_mode: host
    restart: unless-stopped
    environment:
      - IMAGE_URL=http://allskypi5.lan/current/resized/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_PORT=1883
      - MQTT_TOPIC=Astro/SimpleCloudDetect
      - MQTT_USERNAME=your_username
      - MQTT_PASSWORD=your_password
      - DETECT_INTERVAL=60
```

#### With Local Image File

```yaml
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    image: chvvkumar/simpleclouddetect:latest
    network_mode: host
    restart: unless-stopped
    environment:
      - IMAGE_URL=file:///tmp/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_DISCOVERY_MODE=homeassistant
      - DEVICE_ID=clouddetect_001
    volumes:
      - '$HOME/path/to/image.jpg:/tmp/image.jpg'
```

#### With Custom Model

```yaml
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    image: chvvkumar/simpleclouddetect:latest
    network_mode: host
    restart: unless-stopped
    environment:
      - IMAGE_URL=http://allskypi5.lan/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_DISCOVERY_MODE=homeassistant
      - DEVICE_ID=clouddetect_001
    volumes:
      - /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5
      - /docker/simpleclouddetect/labels.txt:/app/labels.txt
```

---

## Home Assistant Integration

### Option 1: MQTT Discovery (Recommended - Zero Configuration)

When using `MQTT_DISCOVERY_MODE=homeassistant`, your device automatically appears in Home Assistant with **no YAML configuration needed**.

**What You Get:**
- Single device with your custom name
- Three sensors:
  - **Cloud Status** - Current sky condition
  - **Confidence** - Detection confidence (%)
  - **Detection Time** - Processing time (seconds)
- Availability tracking (online/offline status)
- Proper device grouping in HA UI

**Setup Steps:**
1. Start container with HA discovery mode (see examples above)
2. In Home Assistant: **Settings → Devices & Services → MQTT**
3. Your cloud detector appears automatically under "MQTT Devices"

> **Tip:** Use unique `DEVICE_ID` values if you have multiple AllSky cameras

### Option 2: Legacy Mode (Manual Configuration)

For custom setups or backward compatibility, configure sensors manually in `configuration.yaml`:

```yaml
mqtt:
  sensor:
    - name: "Cloud Status"
      unique_id: cloud_status_sensor_001
      icon: mdi:clouds
      state_topic: "Astro/SimpleCloudDetect"
      value_template: "{{ value_json.class_name }}"

    - name: "Cloud Status Confidence"
      unique_id: cloud_confidence_sensor_001
      icon: mdi:percent
      state_topic: "Astro/SimpleCloudDetect"
      value_template: "{{ value_json.confidence_score }}"
      unit_of_measurement: "%"

    - name: "Cloud Detection Time"
      unique_id: cloud_detection_time_001
      icon: mdi:timer
      state_topic: "Astro/SimpleCloudDetect"
      value_template: "{{ value_json['Detection Time (Seconds)'] }}"
      unit_of_measurement: "s"
```

---

## ASCOM Alpaca SafetyMonitor

The container includes an ASCOM Alpaca SafetyMonitor service for astronomy automation software.

### Quick Setup

1. **Start Container** with Alpaca environment variables (included in examples above)
2. **Access Setup Page**: `http://<your-server-ip>:11111/setup/v1/safetymonitor/0/setup`
3. **Configure Device**: Set name, location, and unsafe conditions
4. **Add to Software**: Configure in N.I.N.A., SGP, TheSkyX, etc.

### Supported Software

- N.I.N.A. (Nighttime Imaging 'N' Astronomy)
- Sequence Generator Pro
- TheSkyX
- Any ASCOM Alpaca-compatible application

> **Full Documentation**: See **[ALPACA_README.md](ALPACA_README.md)** for detailed configuration, API reference, and troubleshooting.

---

## Manual Installation (Non-Docker)

### Prerequisites

Ensure Python 3.11 is installed:

```shell
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.11 python3.11-venv
```

### Installation Steps

1. **Clone Repository**
```shell
cd ~
mkdir -p git
cd git
git clone https://github.com/chvvkumar/simpleCloudDetect.git
cd simpleCloudDetect
```

2. **Create Virtual Environment**
```shell
python3.11 -m venv env
source env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Configure Settings**

Edit `detect.py` with your settings:
```python
# Define parameters
image_url = "http://localhost/current/resized/image.jpg"
broker = "192.168.1.250"
port = 1883
topic = "Astro/SimpleCloudDetect"
detect_interval = 60
```

4. **Test Detection**
```shell
python3 detect.py
```

### Setup as System Service

Enable automatic startup with systemd:

```shell
sudo cp detect.service /etc/systemd/system/detect.service
sudo systemctl daemon-reload
sudo systemctl enable detect.service
sudo systemctl start detect.service
sudo systemctl status detect.service
```

**Expected Output:**
```shell
● detect.service - Cloud Detection Service
     Loaded: loaded (/etc/systemd/system/detect.service; enabled; preset: enabled)
     Active: active (running) since Sat 2024-10-26 10:08:08 CDT; 5min ago
   Main PID: 5694 (python)
```

---

## Training Your Own Model

While an example model is included, **training your own model with your camera's images is highly recommended** for better accuracy.

### Using Google's Teachable Machine

1. **Go to**: https://teachablemachine.withgoogle.com
2. **Create a New Image Project**
3. **Add Classes**: Clear, Wisps, Mostly Cloudy, Overcast, Rain, Snow
4. **Upload Training Images** from your AllSky camera for each class
5. **Train Model**
6. **Export Model**: Select "TensorFlow" → "Keras" format
7. **Download** both `keras_model.h5` and `labels.txt`

### Training Steps (Visual Guide)

![Step 1](/images/1.png)
![Step 2](/images/2.png)
![Step 3](/images/3.png)
![Step 4](/images/4.png)

### Using Your Custom Model

**For Docker:**
```shell
docker run -d --name simple-cloud-detect --network=host \
  -v /path/to/your/keras_model.h5:/app/keras_model.h5 \
  -v /path/to/your/labels.txt:/app/labels.txt \
  # ...other environment variables...
  chvvkumar/simpleclouddetect:latest
```

**For Manual Installation:**
1. Copy `keras_model.h5` and `labels.txt` to your script directory
2. Convert the model:
```shell
python3 convert.py
```
3. Test with `python3 detect.py`

> **Note:** Docker containers automatically convert the model on startup.

---

## Recent Changes

- **2025-01-30**: Add Home Assistant MQTT Discovery support for automatic device/entity creation
- **2025-01-30**: Add ASCOM Alpaca SafetyMonitor implementation
- **2025-01-09**: Add MQTT authentication support and improved logging
- **2024-12-16**: Add custom model and labels file support via bind mounts
- **2024-11-19**: Add local image file support
- **2024-10-26**: Initial release

---

## Documentation

- **[Main Documentation](readme.md)** - This file
- **[Alpaca SafetyMonitor Guide](ALPACA_README.md)** - ASCOM Alpaca implementation details

---

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/chvvkumar/simpleCloudDetect).
