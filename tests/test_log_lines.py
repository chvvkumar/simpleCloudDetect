import logging
import os
from unittest.mock import patch

from PIL import Image

from detect import CloudDetector, Config


def _config(onnx_model_file, labels_file):
    env = {
        "IMAGE_URL": "file:///" + onnx_model_file,  # placeholder, image loading is patched
        "MODEL_PATH": onnx_model_file,
        "LABEL_PATH": labels_file,
        "DETECT_INTERVAL": "60",
    }
    with patch.dict(os.environ, env, clear=False):
        return Config.from_env()


def test_model_loaded_log_line(onnx_model_file, labels_file, caplog):
    cfg = _config(onnx_model_file, labels_file)
    with caplog.at_level(logging.INFO):
        CloudDetector(cfg, mqtt_client=None)
    assert any(
        f"ONNX model loaded successfully from {onnx_model_file}" in r.getMessage()
        for r in caplog.records
    )


def test_first_detection_log_line(onnx_model_file, labels_file, rgb_image, caplog):
    cfg = _config(onnx_model_file, labels_file)
    detector = CloudDetector(cfg, mqtt_client=None)
    with caplog.at_level(logging.INFO, logger="detect"):
        with patch.object(detector, "_load_image", return_value=rgb_image):
            detector.detect()
            detector.detect()
    messages = [r.getMessage() for r in caplog.records]
    assert sum("First detection complete" in m for m in messages) == 1


def test_return_image_contains_independent_copy(onnx_model_file, labels_file, rgb_image):
    cfg = _config(onnx_model_file, labels_file)
    detector = CloudDetector(cfg, mqtt_client=None)
    with patch.object(detector, "_load_image", return_value=rgb_image):
        result = detector.detect(return_image=True)
    assert "image" in result
    assert isinstance(result["image"], Image.Image)
    assert result["image"] is not rgb_image
