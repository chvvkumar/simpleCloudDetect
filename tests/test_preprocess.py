import os
from unittest.mock import patch

import numpy as np

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


def test_preprocess_shape_and_dtype(onnx_model_file, labels_file, rgb_image):
    detector = _detector(onnx_model_file, labels_file)
    out = detector._preprocess_image(rgb_image)
    assert out.shape == (1, 3, 300, 300)
    assert out.dtype == np.float32


def test_preprocess_uses_imagenet_normalization(onnx_model_file, labels_file):
    detector = _detector(onnx_model_file, labels_file)
    from PIL import Image
    # Pure red image: R=255, G=0, B=0
    arr = np.zeros((300, 300, 3), dtype=np.uint8)
    arr[:, :, 0] = 255
    out = detector._preprocess_image(Image.fromarray(arr, mode="RGB"))
    # Red channel: (1.0 - 0.485) / 0.229
    expected_red = (1.0 - 0.485) / 0.229
    # Green channel: (0.0 - 0.456) / 0.224  -- this is the ImageNet path, NOT keras /127.5-1
    expected_green = (0.0 - 0.456) / 0.224
    assert np.allclose(out[0, 0].mean(), expected_red, atol=1e-4)
    assert np.allclose(out[0, 1].mean(), expected_green, atol=1e-4)
