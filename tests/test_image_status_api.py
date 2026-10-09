"""The external status endpoint exposes image staleness so the dashboard can warn before IsSafe flips."""
import threading
from types import SimpleNamespace

import pytest
from flask import Flask

from alpaca.routes.external_api import external_api_bp, init_external_api
from alpaca.routes.management import image_status_view


IMAGE_STATUS = {
    "check_enabled": True,
    "limit_sec": 600,
    "unchanged_sec": 42.5,
    "changed_since_start": True,
    "hash_missing": False,
    "stale": False,
    "detection_fresh": True,
    "detection_age_sec": 12.0,
}


class FakeMonitor:
    def __init__(self):
        self.alpaca_config = SimpleNamespace(device_number=0)
        self.detection_lock = threading.Lock()
        self.latest_detection = None

    def get_image_status(self):
        return dict(IMAGE_STATUS)

    def is_safe(self):
        return True


def make_client(monitor):
    app = Flask(__name__)
    init_external_api(monitor)
    app.register_blueprint(external_api_bp)
    return app.test_client()


def test_status_includes_image_status():
    client = make_client(FakeMonitor())
    data = client.get("/api/ext/v1/status").get_json()
    assert data["image"] == IMAGE_STATUS
    assert data["is_safe"] is True


GREY = "rgb(100, 116, 139)"
GREEN = "rgb(52, 211, 153)"
AMBER = "rgb(251, 191, 36)"
RED = "rgb(248, 113, 113)"


@pytest.mark.parametrize("status, text, subtext, color", [
    (None, "...", "", GREY),
    ({**IMAGE_STATUS, "detection_fresh": False, "detection_age_sec": 183.0}, "No detection for 3m 3s", "Limit 3m 0s", RED),
    ({**IMAGE_STATUS, "detection_fresh": False, "detection_age_sec": None}, "No detection yet", "Limit 3m 0s", RED),
    ({**IMAGE_STATUS, "check_enabled": False}, "Stale check off", "Limit disabled", GREY),
    ({**IMAGE_STATUS, "hash_missing": True, "unchanged_sec": None, "changed_since_start": False},
     "Stale check off", "No image hash from detector", GREY),
    ({**IMAGE_STATUS, "unchanged_sec": None}, "Waiting for first image", "", GREY),
    (IMAGE_STATUS, "Changed: 42s ago", "Limit 10m 0s", GREEN),
    ({**IMAGE_STATUS, "unchanged_sec": 301}, "Changed: 5m 1s ago", "Limit 10m 0s", AMBER),
    ({**IMAGE_STATUS, "unchanged_sec": 3725, "stale": True}, "STALE: unchanged 1h 2m", "Limit 10m 0s", RED),
    ({**IMAGE_STATUS, "unchanged_sec": None, "changed_since_start": False, "stale": True},
     "STALE: no new frame since start", "Limit 10m 0s", RED),
])
def test_image_status_view(status, text, subtext, color):
    assert image_status_view(status, detection_limit_sec=180) == {"text": text, "subtext": subtext, "color": color}


def test_image_status_view_stale_while_still_safe():
    status = {**IMAGE_STATUS, "unchanged_sec": 601, "stale": True}
    assert image_status_view(status, is_safe=True) == {
        "text": "Changed: 10m 1s ago", "subtext": "Limit 10m 0s, unsafe on next detection", "color": RED}


def test_image_status_view_detection_limit_unknown():
    status = {**IMAGE_STATUS, "detection_fresh": False, "detection_age_sec": 200.0}
    assert image_status_view(status)["subtext"] == ""
