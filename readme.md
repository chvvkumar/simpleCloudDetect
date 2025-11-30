# SimpleCloudDetect

## A simple Machine Learning based Cloud Detection for AllSky Cameras

This project provides ML-based cloud detection for AllSky cameras with two primary outputs:
1. **MQTT Publishing** - Publishes cloud status to Home Assistant or other MQTT subscribers
2. **ASCOM Alpaca SafetyMonitor** - Provides an ASCOM-compliant SafetyMonitor device for astronomy software

## Features

- Machine Learning cloud classification (Clear, Wisps, Mostly Cloudy, Overcast, Rain, Snow)
- MQTT integration for Home Assistant
- ASCOM Alpaca SafetyMonitor for astronomy automation (N.I.N.A., SGP, TheSkyX, etc.)
- Docker support with both services running simultaneously
- Support for both URL-based and local file images
- Custom model support
- Confidence scores and detection timing

## Screenshots

Clear Skies
![alt text](/images/HA2.png)

Majority Clouds

![](/images/Mostly%20Cloudy.png)

Wisps of Clouds

![](/images/wisps.png)

Overcast

![](/images/Overcast.png)

## Docker (preferred method, x86 only at the moment)

DEV Branch

[![dev](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/dev.yml/badge.svg)](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/dev.yml) ![Docker Image Size (dev)](https://img.shields.io/docker/image-size/chvvkumar/simpleclouddetect/dev?style=flat&logo=docker&logoSize=auto)

Main branch

[![main](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/main.yml/badge.svg)](https://github.com/chvvkumar/simpleCloudDetect/actions/workflows/main.yml) ![Docker Image Size (latest)](https://img.shields.io/docker/image-size/chvvkumar/simpleclouddetect/latest?style=flat&logo=docker&logoSize=auto) ![](https://img.shields.io/docker/pulls/chvvkumar/simpleclouddetect?style=flat&logo=docker&label=Pulls) 

## Recent Changes

- **2025-11-30**: Add Home Assistant MQTT Discovery support for automatic device/entity creation
- **2025-11-30**: Add ASCOM Alpaca SafetyMonitor implementation for astronomy software integration
- **2025-01-09**: Add MQTT authentication, improve logging to be more descriptive
- **2024-12-16**: Add ability to provide a custom model file and labels file to the container via bind mounts on the docker host. This allows the user to supply their own trained model and classification labels instead of using the example model in this repo.
- **2024-11-19**: Add ability to use local images via https://github.com/chvvkumar/simpleCloudDetect/pull/8
- **2024-10-26**: Initial release with basic cloud detection functionality

## Documentation

- **[Main Documentation (this file)](readme.md)** - MQTT integration and basic setup
- **[Alpaca SafetyMonitor Guide](ALPACA_README.md)** - ASCOM Alpaca implementation details


### Environment Variables

#### Cloud Detection (Shared)
- `IMAGE_URL` - URL or file path to AllSky camera image (required)
- `MQTT_BROKER` - MQTT broker address (required)
- `MQTT_PORT` - MQTT broker port (default: 1883)
- `MQTT_USERNAME` - MQTT authentication username (optional)
- `MQTT_PASSWORD` - MQTT authentication password (optional)
- `DETECT_INTERVAL` - Detection interval in seconds (default: 60)

#### MQTT Publishing Modes
- `MQTT_DISCOVERY_MODE` - MQTT publishing mode: `legacy` or `homeassistant` (default: `legacy`)
  - **legacy**: Publishes JSON to a single topic (requires `MQTT_TOPIC`)
  - **homeassistant**: Uses HA MQTT Discovery for automatic device/entity creation

**Legacy Mode (Manual YAML Configuration)**
- `MQTT_TOPIC` - MQTT topic for publishing (required for legacy mode)

**Home Assistant Discovery Mode (Automatic Configuration)**
- `DEVICE_ID` - Unique device identifier (required for homeassistant mode)
- `DEVICE_NAME` - Custom device name in Home Assistant (default: "Cloud Detector")
- `MQTT_DISCOVERY_PREFIX` - HA discovery prefix (default: "homeassistant")

#### Alpaca SafetyMonitor (Optional)
- `ALPACA_PORT` - HTTP API port (default: 11111)
- `ALPACA_DEVICE_NUMBER` - Device number (default: 0)
- `ALPACA_UPDATE_INTERVAL` - Update interval in seconds (default: 30)

See [ALPACA_README.md](ALPACA_README.md) for detailed Alpaca configuration.

### Docker Run

```shell
docker pull chvvkumar/simpleclouddetect:latest
```

#### Legacy Mode (Manual YAML Configuration)

```shell
# When using an image from a URL
docker run -d --name simple-cloud-detect --network=host \
  -e IMAGE_URL="http://allskypi5.lan/current/resized/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_PORT="1883" \
  -e MQTT_TOPIC="Astro/SimpleCloudDetect" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  -e DETECT_INTERVAL="60" \
  -e ALPACA_PORT="11111" \
  -e ALPACA_UPDATE_INTERVAL="30" \
  -v /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5 \
  -v /docker/simpleclouddetect/labels.txt:/app/labels.txt \
  chvvkumar/simpleclouddetect:latest
```

#### Home Assistant Discovery Mode (Automatic Configuration)

```shell
# When using an image from a URL with HA Discovery
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
  -e ALPACA_PORT="11111" \
  -e ALPACA_UPDATE_INTERVAL="30" \
  -v /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5 \
  -v /docker/simpleclouddetect/labels.txt:/app/labels.txt \
  chvvkumar/simpleclouddetect:latest
```

**Note**: The container runs both MQTT and Alpaca SafetyMonitor services simultaneously.
### Using Local Image Files

As an alternative, mount the image as a volume and reference it with the `IMAGE_URL` environment variable:

**Legacy Mode:**
```shell
# When using an image from a local file path
docker run -d --name simple-cloud-detect --network=host \
  -v /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5 \
  -v /docker/simpleclouddetect/labels.txt:/app/labels.txt \
  -v $HOME/path/to/image.jpg:/tmp/image.jpg \
  -e IMAGE_URL="file:///tmp/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_PORT="1883" \
  -e MQTT_TOPIC="Astro/SimpleCloudDetect" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  -e DETECT_INTERVAL="60" \
  -e ALPACA_PORT="11111" \
  -e ALPACA_UPDATE_INTERVAL="30" \
  chvvkumar/simpleclouddetect:latest
```

**Home Assistant Discovery Mode:**
```shell
# When using an image from a local file path with HA Discovery
docker run -d --name simple-cloud-detect --network=host \
  -v /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5 \
  -v /docker/simpleclouddetect/labels.txt:/app/labels.txt \
  -v $HOME/path/to/image.jpg:/tmp/image.jpg \
  -e IMAGE_URL="file:///tmp/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_PORT="1883" \
  -e MQTT_DISCOVERY_MODE="homeassistant" \
  -e DEVICE_ID="clouddetect_001" \
  -e DEVICE_NAME="AllSky Cloud Detector" \
  -e MQTT_USERNAME="your_username" \
  -e MQTT_PASSWORD="your_password" \
  -e DETECT_INTERVAL="60" \
  -e ALPACA_PORT="11111" \
  -e ALPACA_UPDATE_INTERVAL="30" \
  chvvkumar/simpleclouddetect:latest
```

### Docker Compose

#### Legacy Mode (Manual YAML Configuration)

```yaml
# When using an image from a URL
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    network_mode: host
    environment:
      - IMAGE_URL=http://allskypi5.lan/current/resized/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_PORT=1883
      - MQTT_TOPIC=Astro/SimpleCloudDetect
      - MQTT_USERNAME=
      - MQTT_PASSWORD=
      - DETECT_INTERVAL=60
      - ALPACA_PORT=11111
      - ALPACA_UPDATE_INTERVAL=30
    volumes:
      - /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5
      - /docker/simpleclouddetect/labels.txt:/app/labels.txt
    restart: unless-stopped
    image: chvvkumar/simpleclouddetect:latest

# When using an image from a local path
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    network_mode: host
    environment:
      - IMAGE_URL=file:///tmp/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_PORT=1883
      - MQTT_TOPIC=Astro/SimpleCloudDetect
      - MQTT_USERNAME=
      - MQTT_PASSWORD=
      - DETECT_INTERVAL=60
      - ALPACA_PORT=11111
      - ALPACA_UPDATE_INTERVAL=30
    volumes:
      - '$HOME/path/to/image.jpg:/tmp/image.jpg'
      - /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5
      - /docker/simpleclouddetect/labels.txt:/app/labels.txt
    restart: unless-stopped
    image: chvvkumar/simpleclouddetect:latest
```

#### Home Assistant Discovery Mode (Automatic Configuration)

```yaml
# When using an image from a URL with HA Discovery
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    network_mode: host
    environment:
      - IMAGE_URL=http://allskypi5.lan/current/resized/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_PORT=1883
      - MQTT_DISCOVERY_MODE=homeassistant
      - DEVICE_ID=clouddetect_001
      - DEVICE_NAME=AllSky Cloud Detector
      - MQTT_USERNAME=
      - MQTT_PASSWORD=
      - DETECT_INTERVAL=60
      - ALPACA_PORT=11111
      - ALPACA_UPDATE_INTERVAL=30
    volumes:
      - /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5
      - /docker/simpleclouddetect/labels.txt:/app/labels.txt
    restart: unless-stopped
    image: chvvkumar/simpleclouddetect:latest

# When using an image from a local path with HA Discovery
services:
  simpleclouddetect:
    container_name: simple-cloud-detect
    network_mode: host
    environment:
      - IMAGE_URL=file:///tmp/image.jpg
      - MQTT_BROKER=192.168.1.250
      - MQTT_PORT=1883
      - MQTT_DISCOVERY_MODE=homeassistant
      - DEVICE_ID=clouddetect_001
      - DEVICE_NAME=AllSky Cloud Detector
      - MQTT_USERNAME=
      - MQTT_PASSWORD=
      - DETECT_INTERVAL=60
      - ALPACA_PORT=11111
      - ALPACA_UPDATE_INTERVAL=30
    volumes:
      - '$HOME/path/to/image.jpg:/tmp/image.jpg'
      - /docker/simpleclouddetect/keras_model.h5:/app/keras_model.h5
      - /docker/simpleclouddetect/labels.txt:/app/labels.txt
    restart: unless-stopped
    image: chvvkumar/simpleclouddetect:latest
```

## ASCOM Alpaca SafetyMonitor

The Docker container also runs an ASCOM Alpaca SafetyMonitor service, allowing astronomy software to automatically monitor sky conditions.

### Quick Start

1. Start the container with Alpaca environment variables (shown above)
2. Access the setup page: `http://<your-server-ip>:11111/setup/v1/safetymonitor/0/setup`
3. Configure device name, location, and unsafe conditions
4. In your ASCOM client (N.I.N.A., SGP, etc.), add a new SafetyMonitor device
5. Use automatic discovery or manually configure with host/port

### Supported Software

- **N.I.N.A.** (Nighttime Imaging 'N' Astronomy)
- **Sequence Generator Pro**
- **TheSkyX**
- **Any ASCOM Alpaca-compatible software**

For detailed Alpaca configuration, API documentation, and troubleshooting, see **[ALPACA_README.md](ALPACA_README.md)**.
---

## Manual Installation (Non-Docker)

### Overview of operations

-   Ensure Python and Python-venv are version 3.11
-   Clone repo
-   Update variables
-   Create venv and activate it
-   Install dependencies from requirements.txt
-   Train your model (my model is included but YMMV with it) and copy to the project firectory
-   Configure your settings for image URL, MQTT server and Home Assistant sensors
-   Verify the output is as expected
-   Set the script to run on boot with cron

## Preperation

If required, install python and python-venv with the correct versions
```shell
sudo  add-apt-repository  ppa:deadsnakes/ppa
sudo  apt  update
sudo  apt  install  python3.11
sudo  apt  install  python3.11-venv
```

Setup folders and venv:
```shell
cd
mkdir git
git clone git@github.com:chvvkumar/simpleCloudDetect.git
cd simpleCloudDetect
python3.11  -m  venv  env && source  env/bin/activate
pip  install  --upgrade  pip
pip  install  -r  requirements.txt
```

Update the script `detect.py` with your own settings for these parameters:
```python
# Define parameters
image_url = "http://localhost/current/resized/image.jpg"
broker = "192.168.1.250"
port = 1883
topic = "Astro/SimpleCloudDetect"
detect_interval = 60
```

## Training your model

The model I use is included in this repo for testing but it is highly recommended to train your own model with your data from your AllSky camera to get a more reliable prediction.

Head on to: 
https://teachablemachine.withgoogle.com

and follow the screenshots to generate a model.

![alt text](/images/1.png)

![alt text](/images/2.png)

![alt text](/images/3.png)

![alt text](/images/4.png)

## Prepare the model for use

- Copy the model file to your script folder where `detect.py` is located 
- Run `python3 convert.py' to convert the model for use
    ```shell
    python3 convert.py
    ```

Run the script to run detection once to test ecverything is working as expected.
```shell
python3  detect.py
```

If using docker, the container takes care of the conversion step automatically.
You only need to mount the model files as a volume:
```shell
docker run -d --name simple-cloud-detect --network=host \
  -v $HOME/path/to/keras_model.h5:/app/keras_model.h5 \
  -v $HOME/path/to/lables.txt:/app/labels.txt \
  -e IMAGE_URL="http://localhost/current/resized/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_PORT="1883" \
  -e MQTT_TOPIC="Astro/SimpleCloudDetect" \
  -e DETECT_INTERVAL="60" \
  chvvkumar/simpleclouddetect:latest
```


## Setup systemd service to automatically start the script on boot and run as a service

Copy the included service file to the systemd folder and enable it
```shell
cd
cd git/simpleCloudDetect
sudo cp detect.service /etc/systemd/system/detect.service
sudo systemctl daemon-reload
sudo systemctl enable detect.service
sudo systemctl start detect.service
sudo systemctl status detect.service
```

Exaample output on successful install

```shell
pi@allskypi5:~/git/simpleCloudDetect $ sudo systemctl status detect.service
● detect.service - Cloud Detection Service
     Loaded: loaded (/etc/systemd/system/detect.service; enabled; preset: enabled)
     Active: active (running) since Sat 2024-10-26 10:08:08 CDT; 5min ago
   Main PID: 5694 (python)
      Tasks: 14 (limit: 4443)
        CPU: 5.493s
     CGroup: /system.slice/detect.service
             └─5694 /home/pi/git/simpleCloudDetect/env/bin/python /home/pi/git/simpleCloudDetect/detect.py

Oct 26 10:10:12 allskypi5 python[5694]: [193B blob data]
Oct 26 10:11:12 allskypi5 python[5694]: Class: Clear Confidence Score: 1.0 Elapsed Time: 0.12
Oct 26 10:11:12 allskypi5 python[5694]: Published data to MQTT topic: Astro/SimpleCloudDetect Data: {"class_name": "Clear", "confidence_score": 100.0, "Detection Time (Seconds)": 0.12}
Oct 26 10:11:12 allskypi5 python[5694]: [193B blob data]
Oct 26 10:12:13 allskypi5 python[5694]: Class: Clear Confidence Score: 1.0 Elapsed Time: 0.11
Oct 26 10:12:13 allskypi5 python[5694]: Published data to MQTT topic: Astro/SimpleCloudDetect Data: {"class_name": "Clear", "confidence_score": 100.0, "Detection Time (Seconds)": 0.11}
Oct 26 10:12:13 allskypi5 python[5694]: [193B blob data]
Oct 26 10:13:13 allskypi5 python[5694]: Class: Clear Confidence Score: 1.0 Elapsed Time: 0.11
Oct 26 10:13:13 allskypi5 python[5694]: Published data to MQTT topic: Astro/SimpleCloudDetect Data: {"class_name": "Clear", "confidence_score": 100.0, "Detection Time (Seconds)": 0.11}
Oct 26 10:13:13 allskypi5 python[5694]: [193B blob data]
```

## Home Assistant Integration

### Option 1: Home Assistant MQTT Discovery (Recommended)

**No configuration required!** When using `MQTT_DISCOVERY_MODE=homeassistant`, the device and all sensors automatically appear in Home Assistant.

**What you get:**
- A single device named according to your `DEVICE_NAME` setting
- Three sensors automatically created:
  - **Cloud Status** - Current sky condition (Clear, Wisps, Mostly Cloudy, Overcast, Rain, Snow)
  - **Confidence** - Detection confidence percentage
  - **Detection Time** - Time taken for detection in seconds
- Availability tracking - Shows when the detector is online/offline
- Proper device grouping in Home Assistant UI

**Setup:**
1. Configure your container with HA discovery mode (see examples above)
2. Start the container
3. In Home Assistant, go to **Settings → Devices & Services → MQTT**
4. Your cloud detector device will appear automatically under "MQTT Devices"

**Device ID:** Choose a unique `DEVICE_ID` for each detector if you have multiple AllSky cameras.

### Option 2: Legacy Mode (Manual YAML Configuration)

For backward compatibility or custom setups, use legacy mode with manual sensor configuration.

Add this to your MQTT sensor configuration (`configuration.yaml`):
```yaml
mqtt:
  sensor:
    - name: "Cloud Status"
      unique_id: DXWiwkjvhjhzf7KGwAFDAo7K
      icon: mdi:clouds
      state_topic: "Astro/SimpleCloudDetect"
      value_template: "{{ value_json.class_name }}"

    - name: "Cloud Status Confidence"
      unique_id: tdrgfwkjvhjhzf7KGwAFDAo7K
      icon: mdi:percent
      state_topic: "Astro/SimpleCloudDetect"
      value_template: "{{ value_json.confidence_score }}"
      unit_of_measurement: "%"

    - name: "Cloud Detection Time"
      unique_id: dfhfyuyjghjhcjzf7
      icon: mdi:timer
      state_topic: "Astro/SimpleCloudDetect"
      value_template: "{{ value_json['Detection Time (Seconds)'] }}"
      unit_of_measurement: "s"
```
