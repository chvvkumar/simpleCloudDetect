import os
from unittest.mock import patch

from detect import CloudDetector, Config


def _detector(onnx_model_file, labels_file):
    env = {
        "IMAGE_URL": "http://example.com/image.jpg",
        "MODEL_PATH": onnx_model_file,
        "LABEL_PATH": labels_file,
        "DETECT_INTERVAL": "60",
    }
    with patch.dict(os.environ, env, clear=False):
        cfg = Config.from_env()
    return CloudDetector(cfg, mqtt_client=None)


def test_prefixed_label_is_stripped(onnx_model_file, labels_file, rgb_image):
    # The tiny model's bias makes argmax == index 5 deterministically.
    detector = _detector(onnx_model_file, labels_file)
    detector.class_names = [
        "0 Clear", "1 Mostly Cloudy", "2 Overcast",
        "3 Partly Cloudy", "4 Rain", "5 Snow",
    ]
    with patch.object(detector, "_load_image", return_value=rgb_image):
        result = detector.detect()
    assert result["class_name"] == "Snow"


def test_plain_label_passthrough(onnx_model_file, labels_file, rgb_image):
    detector = _detector(onnx_model_file, labels_file)
    detector.class_names = [
        "Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Rain", "Snow",
    ]
    with patch.object(detector, "_load_image", return_value=rgb_image):
        result = detector.detect()
    assert result["class_name"] == "Snow"


def test_out_of_range_index_maps_to_unknown(onnx_model_file, labels_file, rgb_image):
    detector = _detector(onnx_model_file, labels_file)
    # Only 3 labels but argmax == 5 -> index out of range -> "Unknown"
    detector.class_names = ["Clear", "Mostly Cloudy", "Overcast"]
    with patch.object(detector, "_load_image", return_value=rgb_image):
        result = detector.detect()
    assert result["class_name"] == "Unknown"
