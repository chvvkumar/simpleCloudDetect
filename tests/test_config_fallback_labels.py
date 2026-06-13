import importlib
import os
from unittest.mock import patch


def test_fallback_labels_are_six_class_taxonomy(tmp_path):
    missing = str(tmp_path / "does_not_exist.txt")
    with patch.dict(os.environ, {"LABEL_PATH": missing}, clear=False):
        import alpaca.config as cfg
        importlib.reload(cfg)
        assert cfg.load_labels() == [
            "Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Rain", "Snow"
        ]
