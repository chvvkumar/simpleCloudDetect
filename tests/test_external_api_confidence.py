"""Tests for the confidence-stats external API routes using a fake monitor."""
import pytest
from flask import Flask
from alpaca.routes.external_api import external_api_bp, init_external_api
from alpaca.confidence_stats import ConfidenceStats


class FakeMonitor:
    def __init__(self, stats):
        self.confidence_stats = stats


@pytest.fixture
def client(tmp_path):
    stats = ConfidenceStats(filepath=str(tmp_path / "cs.json"), classes=["Clear", "Rain"])
    stats.record("Rain", 42.0)
    app = Flask(__name__)
    init_external_api(FakeMonitor(stats))
    app.register_blueprint(external_api_bp)
    return app.test_client(), stats


def test_get_confidence_stats(client):
    test_client, _ = client
    resp = test_client.get("/api/ext/v1/confidence-stats")
    assert resp.status_code == 200

    data = resp.get_json()
    for key in ("version", "since", "total_detections", "classes", "advice"):
        assert key in data

    assert data["total_detections"] == 1

    names = [c["name"] for c in data["classes"]]
    assert "Clear" in names
    assert "Rain" in names


def test_rain_entry_count_and_recommendation(client):
    test_client, _ = client
    resp = test_client.get("/api/ext/v1/confidence-stats")
    assert resp.status_code == 200

    data = resp.get_json()
    rain = next(c for c in data["classes"] if c["name"] == "Rain")
    assert rain["count"] == 1
    assert "severity" in rain["recommendation"]


def test_reset_confidence_stats(client):
    test_client, _ = client

    resp = test_client.post("/api/ext/v1/confidence-stats/reset")
    assert resp.status_code == 200

    resp = test_client.get("/api/ext/v1/confidence-stats")
    assert resp.status_code == 200

    data = resp.get_json()
    assert data["total_detections"] == 0
    for c in data["classes"]:
        assert c["count"] == 0
