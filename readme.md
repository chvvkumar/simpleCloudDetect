<div align="center">

## ☁️ SimpleCloudDetect


A Machine Learning-based cloud detection system for AllSky cameras with MQTT and ASCOM Alpaca SafetyMonitor integration.

[![Main Build](https://img.shields.io/github/actions/workflow/status/chvvkumar/simpleCloudDetect/build-and-release.yml?branch=main&label=main&logo=github)](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/build-and-release.yml) [![Dev Build](https://img.shields.io/github/actions/workflow/status/chvvkumar/simpleCloudDetect/build-and-release.yml?branch=dev&label=dev&logo=github)](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/build-and-release.yml?query=branch%3Adev)

[![Docker Image Size (latest)](https://img.shields.io/docker/image-size/chvvkumar/simpleclouddetect/latest?style=flat&logo=docker&logoSize=auto)](https://hub.docker.com/r/chvvkumar/simpleclouddetect) [![Docker Pulls](https://img.shields.io/docker/pulls/chvvkumar/simpleclouddetect?style=flat&logo=docker&label=Pulls)](https://hub.docker.com/r/chvvkumar/simpleclouddetect) 


[![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat&logo=python&logoColor=white)](https://www.python.org/) [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange?style=flat&logo=tensorflow)](https://www.tensorflow.org/) [![Home Assistant](https://img.shields.io/badge/Home%20Assistant-Compatible-blue?style=flat&logo=home-assistant)](https://www.home-assistant.io/) [![ASCOM](https://img.shields.io/badge/ASCOM-Alpaca-green?style=flat)](https://ascom-standards.org/)
</div>

---

## Table of Contents

- [Features](#features)
- [Quick Start](#quick-start)
- [Screenshots](#screenshots)
- [Docker Installation](#docker-installation)
  - [Environment Variables](#environment-variables)
  - [Docker Run Examples](#docker-run-examples)
  - [Docker Compose Examples](#docker-compose-examples)
- [Home Assistant Integration](#home-assistant-integration)
- [ASCOM Alpaca SafetyMonitor](#ascom-alpaca-safetymonitor)
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
  -v /path/to/keras_model.h5:/app/keras_model.h5 \
  -v /path/to/labels.txt:/app/labels.txt \
  -e IMAGE_URL="http://your-allsky-camera/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DETECT_INTERVAL="60" \
  -e VERIFY_SSL="false" \
  -e DEVICE_ID="clouddetect_001" \
  chvvkumar/simpleclouddetect:latest
```

> **Important:** The model files (`keras_model.h5` and `labels.txt`) must have read+write permissions for the container user. To ensure proper permissions:
> 
> ```shell
> # Set appropriate permissions (Linux/macOS)
> chmod 666 /path/to/keras_model.h5
> chmod 666 /path/to/labels.txt
> 
> # Or set ownership to your user and make group-writable
> chown $USER:$USER /path/to/keras_model.h5 /path/to/labels.txt
> chmod 664 /path/to/keras_model.h5 /path/to/labels.txt
> ```

That's it! Your device will automatically appear in Home Assistant under **Settings → Devices & Services → MQTT**.

---

## Screenshots

### Cloud Detection Examples

| Condition | Example |
|-----------|---------|
| **Clear Skies** | ![Clear](/images/HA2.png) |
| **Majority Clouds** | ![Mostly Cloudy](/images/Mostly%20Cloudy.png) |
| **Wisps of Clouds** | ![Wisps](/images/wisps.png) |
| **Overcast** | ![Overcast](/images/Overcast.png) |

### ASCOM Alpaca Settings Interface

| View | Screenshot |
|------|------------|
| **Settings Collapsed** | ![Settings Collapsed](/images/Settings_Collapsed.png) |
| **Settings Expanded** | ![Settings Expanded](/images/Settings_Expanded.png) |

---

## Docker Installation

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
| `VERIFY_SSL` | `false` | Set to `true` to enable SSL certificate verification for HTTPS `IMAGE_URL`s. Defaults to `false` for convenience with self-signed certificates. |

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

### Raspberry Pi Support

**Multi-architecture support:** The Docker images are built for both AMD64 (x86_64) and ARM64 (Raspberry Pi 4/5). Docker will automatically pull the correct image for your platform.

For Raspberry Pi, use the same docker commands. The ARM64 build uses full TensorFlow instead of tensorflow-cpu for compatibility.

> **Note:** First run on Raspberry Pi may take longer as it downloads the ARM64 image (~500MB).

### Docker Run Examples

#### Home Assistant Discovery Mode (Recommended)

**With URL-based image:**
```shell
docker run -d --name simple-cloud-detect --network=host \
  -v /path/to/keras_model.h5:/app/keras_model.h5 \
  -v /path/to/labels.txt:/app/labels.txt \
  -e IMAGE_URL="http://allskypi5.lan/current/resized/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_PORT="1883" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DEVICE_ID="clouddetect_001" \
  -e DEVICE_NAME="AllSky Cloud Detector" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  -e DETECT_INTERVAL="60" \
  -e VERIFY_SSL="false" \
  chvvkumar/simpleclouddetect:latest
```

**With local file:**
```shell
docker run -d --name simple-cloud-detect --network=host \
  -v $HOME/path/to/image.jpg:/tmp/image.jpg \
  -v /path/to/keras_model.h5:/app/keras_model.h5 \
  -v /path/to/labels.txt:/app/labels.txt \
  -e IMAGE_URL="file:///tmp/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DETECT_INTERVAL="60" \
  -e DEVICE_ID="clouddetect_001" \
  chvvkumar/simpleclouddetect:latest
```

#### Legacy Mode

**With URL-based image:**
```shell
docker run -d --name simple-cloud-detect --network=host \
  -v /path/to/keras_model.h5:/app/keras_model.h5 \
  -v /path/to/labels.txt:/app/labels.txt \
  -e IMAGE_URL="http://allskypi5.lan/current/resized/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_TOPIC="Astro/SimpleCloudDetect" \
  -e DETECT_INTERVAL="60" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  -e VERIFY_SSL="false" \
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
  -e DETECT_INTERVAL="60" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DEVICE_ID="clouddetect_001" \
  -e VERIFY_SSL="false" \
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
      - IMAGE_URL=http://localhost/current/resized/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_PORT=1883
      - MQTT_DISCOVERY_MODE=homeassistant
      - DEVICE_ID=clouddetect001
      - DEVICE_NAME=AllSkyPi5 Cloud Detector
      - MQTT_USERNAME=
      - MQTT_PASSWORD=
      - DETECT_INTERVAL=60
      - VERIFY_SSL=false
      - ALPACA_PORT=11111
      - ALPACA_UPDATE_INTERVAL=30
    volumes:
      - /home/pi/git/simpleCloudDetect/keras_model.h5:/app/keras_model.h5
      - /home/pi/git/simpleCloudDetect/labels.txt:/app/labels.txt
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
      - VERIFY_SSL=false
    volumes:
      - /home/pi/git/simpleCloudDetect/keras_model.h5:/app/keras_model.h5
      - /home/pi/git/simpleCloudDetect/labels.txt:/app/labels.txt
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
      - /home/pi/git/simpleCloudDetect/keras_model.h5:/app/keras_model.h5
      - /home/pi/git/simpleCloudDetect/labels.txt:/app/labels.txt
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

![Home Assistant MQTT Integration](/images/HA_MQTT.jpg)

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

The container includes an ASCOM Alpaca SafetyMonitor service (Interface Version 3) for astronomy automation software.

### Quick Setup

1. **Start Container** with Alpaca environment variables (included in examples above)
2. **Access Setup Page**: `http://<your-server-ip>:11111/setup/v1/safetymonitor/0/setup`
3. **Configure Device**: Set name, location, and unsafe conditions
4. **Add to Software**: Configure in N.I.N.A., SGP, TheSkyX, etc.

![Alpaca Setup Page](/images/setup.jpg)

### Supported Software

- N.I.N.A. (Nighttime Imaging 'N' Astronomy)
- Sequence Generator Pro
- TheSkyX
- Any ASCOM Alpaca-compatible application

![N.I.N.A. Integration](/images/NINA.jpg)

> **Full Documentation**: See **[ALPACA_README.md](ALPACA_README.md)** for detailed configuration, API reference, and troubleshooting.

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

> **Note:** Docker containers automatically convert the model on startup.

---

## Recent Changes

- **2025-12-30**: Add `VERIFY_SSL` environment variable to disable SSL certificate verification for HTTPS `IMAGE_URL`s. Defaults to `false`.
- **2024-01-30**: Add multi-arch support with support for ARM (Raspberry Pi)
- **2024-01-30**: Add Home Assistant MQTT Discovery support for automatic device/entity creation
- **2024-01-30**: Add ASCOM Alpaca SafetyMonitor implementation
- **2024-01-09**: Add MQTT authentication support and improved logging
- **2024-12-16**: Add custom model and labels file support via bind mounts
- **2024-11-19**: Add local image file support
- **2024-10-26**: Initial release

---

## Documentation

- **[Main Documentation](readme.md)** - This file
- **[System Architecture](architecture.md)** - A detailed look at the components and data flow.
- **[Alpaca SafetyMonitor Guide](ALPACA_README.md)** - ASCOM Alpaca implementation details

---

## Support

For issues, questions, or contributions, please visit the [GitHub repository](https://github.com/chvvkumar/simpleCloudDetect).
