"""Unit tests for alpaca.confidence_stats.ConfidenceStats."""
import json
import threading

import pytest

from alpaca.confidence_stats import (
    ConfidenceStats,
    _bucket_index,
    NUM_BUCKETS,
    MIN_SAMPLES,
)

CLASSES = ["Clear", "Mostly Cloudy", "Overcast", "Partly Cloudy", "Rain", "Snow"]


@pytest.fixture
def stats_path(tmp_path):
    return str(tmp_path / "confidence_stats.json")


@pytest.fixture
def stats(stats_path):
    return ConfidenceStats(filepath=stats_path, classes=CLASSES)


def test_bucket_boundaries():
    assert _bucket_index(0) == 0
    assert _bucket_index(9.9) == 0
    assert _bucket_index(10) == 1
    assert _bucket_index(55.0) == 5
    assert _bucket_index(89.9) == 8
    assert _bucket_index(90) == 9
    assert _bucket_index(100) == NUM_BUCKETS - 1  # closed at 100
    # Out-of-range values are clamped.
    assert _bucket_index(-5) == 0
    assert _bucket_index(150) == NUM_BUCKETS - 1


def test_preseeds_known_classes(stats):
    summary = stats.summary()
    names = [c["name"] for c in summary["classes"]]
    assert names == CLASSES
    assert summary["total_detections"] == 0
    for c in summary["classes"]:
        assert c["count"] == 0
        assert c["recommendation"]["severity"] == "none"


def test_record_updates_count_mean_minmax_buckets(stats):
    stats.record("Rain", 40.0)
    stats.record("Rain", 60.0)
    stats.record("Rain", 50.0)

    summary = stats.summary()
    rain = next(c for c in summary["classes"] if c["name"] == "Rain")
    assert rain["count"] == 3
    assert rain["mean"] == 50.0
    assert rain["min"] == 40.0
    assert rain["max"] == 60.0
    # 40 -> bin 4, 60 -> bin 6, 50 -> bin 5
    assert rain["buckets"][4] == 1
    assert rain["buckets"][5] == 1
    assert rain["buckets"][6] == 1
    assert summary["total_detections"] == 3


def test_on_demand_class_creation(stats):
    stats.record("Fog", 75.0)  # not in the initial label set
    summary = stats.summary()
    names = [c["name"] for c in summary["classes"]]
    assert "Fog" in names
    fog = next(c for c in summary["classes"] if c["name"] == "Fog")
    assert fog["count"] == 1


def test_invalid_inputs_ignored(stats):
    stats.record("", 50.0)
    stats.record(None, 50.0)
    stats.record("Rain", "not-a-number")
    assert stats.summary()["total_detections"] == 0


def test_reset_clears_state_and_updates_since(stats):
    stats.record("Rain", 40.0)
    before = stats.summary()["since"]
    stats.reset()
    after = stats.summary()
    assert after["total_detections"] == 0
    assert all(c["count"] == 0 for c in after["classes"])
    # Known classes are retained after reset.
    assert [c["name"] for c in after["classes"]] == CLASSES
    assert after["since"] >= before


def test_persistence_round_trip(stats, stats_path):
    for _ in range(3):
        stats.record("Snow", 80.0)
    # File exists and is valid JSON.
    with open(stats_path) as f:
        on_disk = json.load(f)
    assert on_disk["total_detections"] == 3

    # A fresh instance pointed at the same file restores state.
    reloaded = ConfidenceStats(filepath=stats_path, classes=CLASSES)
    snow = next(c for c in reloaded.summary()["classes"] if c["name"] == "Snow")
    assert snow["count"] == 3
    assert snow["mean"] == 80.0


def test_recommendation_low_confidence(stats):
    for _ in range(MIN_SAMPLES + 10):
        stats.record("Rain", 30.0)  # well below LOW_MEAN_PCT
    rain = next(c for c in stats.summary()["classes"] if c["name"] == "Rain")
    assert rain["recommendation"]["severity"] == "warn"


def test_recommendation_too_few_samples(stats):
    stats.record("Rain", 95.0)
    rain = next(c for c in stats.summary()["classes"] if c["name"] == "Rain")
    assert rain["recommendation"]["severity"] == "info"


def test_recommendation_healthy(stats):
    for _ in range(MIN_SAMPLES + 10):
        stats.record("Clear", 95.0)
    clear = next(c for c in stats.summary()["classes"] if c["name"] == "Clear")
    assert clear["recommendation"]["severity"] == "ok"


def test_advice_list_mentions_flagged_classes(stats):
    for _ in range(MIN_SAMPLES + 10):
        stats.record("Rain", 30.0)
    advice = stats.summary()["advice"]
    assert any(a.startswith("Rain:") for a in advice)


def test_concurrent_record_is_thread_safe(stats):
    def worker():
        for _ in range(100):
            stats.record("Clear", 70.0)

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    clear = next(c for c in stats.summary()["classes"] if c["name"] == "Clear")
    assert clear["count"] == 500
    assert stats.summary()["total_detections"] == 500
