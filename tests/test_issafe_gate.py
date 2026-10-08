"""IsSafe is gated per Alpaca client; the external API reports raw detector state."""
import threading
from types import SimpleNamespace

import pytest
from flask import Flask, request

from alpaca.routes.api import api_bp, init_api
from alpaca.routes.external_api import external_api_bp, init_external_api


class FakeMonitor:
    def __init__(self, safe=True):
        self.alpaca_config = SimpleNamespace(device_number=0)
        self._safe = safe
        self.connected = set()
        self.detection_lock = threading.Lock()
        self.latest_detection = None

    def get_client_params(self):
        return int(request.args.get("ClientID", 0)), int(request.args.get("ClientTransactionID", 0))

    def register_heartbeat(self, ip, client_id):
        pass

    def create_response(self, value=None, error_number=0, error_message="", client_transaction_id=0):
        return {
            "Value": value,
            "ErrorNumber": error_number,
            "ErrorMessage": error_message,
            "ClientTransactionID": client_transaction_id,
        }

    def is_client_connected(self, ip, client_id):
        return (ip, client_id) in self.connected

    def is_safe(self):
        return self._safe


@pytest.fixture
def harness():
    monitor = FakeMonitor(safe=True)
    app = Flask(__name__)
    init_api(monitor)
    init_external_api(monitor)
    app.register_blueprint(api_bp, url_prefix="/api")
    app.register_blueprint(external_api_bp)
    return app.test_client(), monitor


def test_issafe_false_for_unconnected_client(harness):
    client, _ = harness
    resp = client.get("/api/v1/safetymonitor/0/issafe?ClientID=5")
    assert resp.get_json()["Value"] is False


def test_issafe_true_for_connected_client_only(harness):
    client, monitor = harness
    monitor.connected.add(("127.0.0.1", 5))
    assert client.get("/api/v1/safetymonitor/0/issafe?ClientID=5").get_json()["Value"] is True
    assert client.get("/api/v1/safetymonitor/0/issafe?ClientID=6").get_json()["Value"] is False


def test_devicestate_uses_same_gate(harness):
    client, monitor = harness
    assert client.get("/api/v1/safetymonitor/0/devicestate?ClientID=5").get_json()["Value"] == [
        {"Name": "IsSafe", "Value": False}
    ]
    monitor.connected.add(("127.0.0.1", 5))
    assert client.get("/api/v1/safetymonitor/0/devicestate?ClientID=5").get_json()["Value"] == [
        {"Name": "IsSafe", "Value": True}
    ]


def test_external_status_ignores_client_connections(harness):
    client, _ = harness
    data = client.get("/api/ext/v1/status").get_json()
    assert data["is_safe"] is True
    assert data["safety_status"] == "Safe"
