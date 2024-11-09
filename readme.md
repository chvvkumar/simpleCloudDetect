

## A simple Machine Learning based Cloud Detection for AllSky Cameras

  This python script will take an image from an AllSky camera, run it through a machine learning model and output cloud status to Home Assistant

## Screenshots

Clear Skies
![alt text](/images/HA2.png)

Majority Clouds

![](/images/Mostly%20Cloudy.png)

Wisps of Clouds

![](/images/wisps.png)

Overcast

![](/images/Overcast.png)

## Docker (preferred method)

![Docker build status](https://img.shields.io/github/actions/workflow/status/chvvkumar/simpleclouddetect/main.yml
) ![](https://img.shields.io/docker/image-size/chvvkumar/simpleclouddetect
) ![](https://img.shields.io/docker/pulls/chvvkumar/simpleclouddetect
) 

docker run:
```shell
docker pull chvvkumar/simpleclouddetect:latest

docker run -d --name simple-cloud-detect --network=host \
  -e IMAGE_URL="http://localhost/current/resized/image.jpg" \
  -e MQTT_BROKER="192.168.1.250" \
  -e MQTT_PORT="1883" \
  -e MQTT_TOPIC="Astro/SimpleCloudDetect" \
  -e DETECT_INTERVAL="60" \
  chvvkumar/simpleclouddetect:latest \
```
docker compose:

```shell
    simpleclouddetect:
        container_name: simple-cloud-detect
        network_mode: host
        environment:
            - IMAGE_URL=http://localhost/current/resized/image.jpg
            - MQTT_BROKER=192.168.1.250
            - MQTT_PORT=1883
            - MQTT_TOPIC=Astro/SimpleCloudDetect
            - DETECT_INTERVAL=15
        image: chvvkumar/simpleclouddetect:latest
```
## Manual install and run Overview of operations

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

## Add sensors to Home Assistant

Add this to your MQTT sensor configuration
```yaml
- name: "Cloud Status"
    unique_id: DXWiwkjvhjhzf7KGwAFDAo7K
    icon: mdi:clouds
    state_topic: "Astro/Skytatus"
    value_template: "{{ value_json.class_name }}"

- name: "Cloud Status Confidence"
    unique_id: tdrgfwkjvhjhzf7KGwAFDAo7K
    icon: mdi:exclamation
    state_topic: "Astro/Skytatus"
    value_template: "{{ value_json.confidence_score | float * 100 }}"
    unit_of_measurement: "%"

- name: "Cloud Detection Time"
    unique_id: dfhfyuyjghjhcjzf7
    icon: mdi:exclamation
    state_topic: "Astro/SimpleCloudDetect"
    value_template: "{{ value_json['Detection Time (Seconds)'] }}"
    unit_of_measurement: "S"

```