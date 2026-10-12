"""IsSafe must fail safe when the source image stops changing or detection keeps failing."""
import hashlib
import json
import types
from datetime import datetime, timedelta, timezone

import pytest

import alpaca.device as device
from alpaca.config import AlpacaConfig


class FakeDetector:
    def __init__(self, *a, **k):
        self.image_hash = 'a'
        self.fail = False

    def detect(self, return_image=False):
        if self.fail:
            raise IOError("image download failed")
        return {'class_name': 'Clear', 'confidence_score': 99.0,
                'Detection Time (Seconds)': 0.1, 'image_hash': self.image_hash}


@pytest.fixture
def harness(monkeypatch):
    clock = [datetime(2026, 1, 1, tzinfo=timezone.utc)]
    monkeypatch.setattr(device, 'get_current_time', lambda tz='UTC': clock[0])
    monkeypatch.setattr(device, 'CloudDetector', FakeDetector)
    monkeypatch.setattr(device.AlpacaSafetyMonitor, '_setup_mqtt', lambda self: None)

    def make(max_image_age_sec=600):
        cfg = AlpacaConfig(unsafe_conditions=[], debounce_to_safe_sec=60, debounce_to_unsafe_sec=0,
                           update_interval=30, max_image_age_sec=max_image_age_sec)
        monitor = device.AlpacaSafetyMonitor(cfg, types.SimpleNamespace(broker=None))
        return monitor

    def advance(monitor, seconds, step=30, new_hash=False):
        for _ in range(seconds // step):
            clock[0] += timedelta(seconds=step)
            if new_hash:
                monitor.cloud_detector.image_hash = str(clock[0])
            monitor._run_single_detection()

    return make, advance, clock


def test_fresh_changing_image_is_safe(harness):
    make, advance, _ = harness
    monitor = make()
    assert monitor.is_safe() is False  # startup: debounce not yet elapsed
    advance(monitor, 120, new_hash=True)
    assert monitor.is_safe() is True
    assert 'image_hash' not in monitor.latest_detection


def test_unchanged_image_forces_unsafe_and_recovers_after_debounce(harness):
    make, advance, _ = harness
    monitor = make()
    advance(monitor, 120, new_hash=True)
    assert monitor.is_safe() is True

    advance(monitor, 600)  # hash unchanged for exactly the limit: still safe
    assert monitor.is_safe() is True
    advance(monitor, 30)  # first stale reading commits unsafe immediately, no debounce
    assert monitor.is_safe() is False
    assert monitor.get_safety_history()[-1]['condition'] == 'Stale Image'

    advance(monitor, 30, new_hash=True)  # image changes again: debounce still applies
    assert monitor.is_safe() is False
    advance(monitor, 60, new_hash=True)
    assert monitor.is_safe() is True


def test_stale_image_does_not_start_pending_safe_timer(harness):
    make, advance, _ = harness
    monitor = make()
    advance(monitor, 120, new_hash=True)
    advance(monitor, 660)  # stale
    assert monitor.is_safe() is False
    assert monitor.get_pending_status()['is_pending'] is False
    advance(monitor, 30, new_hash=True)  # recovery starts the safe debounce only now
    assert monitor.get_pending_status()['is_pending'] is True


def test_zero_disables_image_check(harness):
    make, advance, _ = harness
    monitor = make(max_image_age_sec=0)
    advance(monitor, 3600)  # hash never changes since start: fine with the check off
    assert monitor.is_safe() is True


def test_startup_unchanged_hash_never_safe(harness):
    make, advance, _ = harness
    monitor = make()
    advance(monitor, 120)  # past the 60 s safe debounce, inside the 600 s limit: only the startup rule holds
    assert monitor.is_safe() is False
    advance(monitor, 3600)  # hash identical to the first one since boot, well past debounce
    assert monitor.is_safe() is False
    assert monitor.get_pending_status()['is_pending'] is False
    advance(monitor, 90, new_hash=True)  # first change plus 60 s debounce
    assert monitor.is_safe() is True


def test_missing_hash_disables_check_and_warns_once(harness, caplog):
    make, advance, _ = harness
    monitor = make()
    monitor.cloud_detector.image_hash = None
    with caplog.at_level('WARNING', logger='alpaca.device'):
        advance(monitor, 3600)
    assert monitor.is_safe() is True
    assert sum('image_hash missing' in r.message for r in caplog.records) == 1


def test_get_image_status_matches_demote_state(harness):
    make, advance, _ = harness
    monitor = make()
    base = {'check_enabled': True, 'limit_sec': 600, 'hash_missing': False,
            'detection_fresh': True, 'detection_age_sec': 0.0}
    assert monitor.get_image_status() == {**base, 'unchanged_sec': None, 'changed_since_start': False, 'stale': True}
    advance(monitor, 120, new_hash=True)
    assert monitor.get_image_status() == {**base, 'unchanged_sec': 0.0, 'changed_since_start': True, 'stale': False}
    advance(monitor, 630)
    status = monitor.get_image_status()
    assert status['unchanged_sec'] == 630.0 and status['stale'] is True
    assert status['stale'] is not monitor.is_safe()

    monitor.cloud_detector.image_hash = None
    advance(monitor, 30)
    status = monitor.get_image_status()
    assert status['stale'] is False and status['hash_missing'] is True  # check skipped, IsSafe not demoted

    monitor.cloud_detector.fail = True
    advance(monitor, 210)
    status = monitor.get_image_status()
    assert status['detection_fresh'] is False and status['detection_age_sec'] == 210.0

    disabled = make(max_image_age_sec=0)
    advance(disabled, 3600)
    assert disabled.get_image_status() == {**base, 'check_enabled': False, 'limit_sec': 0, 'unchanged_sec': None,
                                           'changed_since_start': False, 'stale': False}


def test_enabling_check_at_runtime_keeps_hash_history(harness):
    make, advance, _ = harness
    monitor = make(max_image_age_sec=0)
    advance(monitor, 120, new_hash=True)
    monitor.alpaca_config.max_image_age_sec = 600
    advance(monitor, 30)  # one unchanged frame: history shows a change already happened
    assert monitor.is_safe() is True


class FakeHA:
    def __init__(self):
        self.calls = []

    def publish_data_availability(self, online):
        self.calls.append(('data', online))

    def publish_safe(self, is_safe):
        self.calls.append(('safe', is_safe))


def test_ha_data_availability_and_safe_follow_demote_and_recovery(harness):
    make, advance, _ = harness
    monitor = make()
    monitor.ha_discovery = FakeHA()
    calls = monitor.ha_discovery.calls

    advance(monitor, 120, new_hash=True)
    assert calls[-2:] == [('data', True), ('safe', True)]
    advance(monitor, 630)  # stale image
    assert calls[-2:] == [('data', False), ('safe', False)]
    advance(monitor, 30, new_hash=True)  # changed again: data current, safe still debouncing
    assert calls[-2:] == [('data', True), ('safe', False)]
    advance(monitor, 60, new_hash=True)
    assert calls[-2:] == [('data', True), ('safe', True)]

    monitor.cloud_detector.fail = True
    del calls[:]
    advance(monitor, 180)  # failing but still within the detection limit: nothing published
    assert calls == []
    advance(monitor, 30)
    assert calls == [('data', False), ('safe', False)]


def test_ha_discovery_payloads():
    from unittest import mock
    from detect import HADiscoveryManager
    client = mock.MagicMock()
    cfg = types.SimpleNamespace(device_id='dev', mqtt_discovery_prefix='homeassistant', device_name='Cloud')
    HADiscoveryManager(cfg, client).publish_discovery_configs()  # standalone detect.py: no binary sensor
    assert not any('binary_sensor' in c.args[0] for c in client.publish.call_args_list)
    client.publish.reset_mock()
    ha = HADiscoveryManager(cfg, client, safe_sensor=True)
    ha.publish_discovery_configs()
    published = {c.args[0]: c.args[1] for c in client.publish.call_args_list}

    status = json.loads(published['homeassistant/sensor/clouddetect_dev/status/config'])
    assert status['availability'] == [{'topic': 'homeassistant/sensor/clouddetect_dev/availability'},
                                      {'topic': 'homeassistant/sensor/clouddetect_dev/data_availability'}]
    assert status['availability_mode'] == 'all'
    safe = json.loads(published['homeassistant/binary_sensor/clouddetect_dev/safe/config'])
    assert safe['state_topic'] == 'homeassistant/binary_sensor/clouddetect_dev/safe/state'
    assert safe['availability'] == [{'topic': 'homeassistant/sensor/clouddetect_dev/availability'}]
    assert 'device_class' not in safe
    assert published['homeassistant/sensor/clouddetect_dev/availability'] == 'online'
    assert published['homeassistant/sensor/clouddetect_dev/data_availability'] == 'online'

    ha.publish_data_availability(False)
    ha.publish_safe(True)
    client.publish.assert_any_call('homeassistant/sensor/clouddetect_dev/data_availability', 'offline', retain=True)
    client.publish.assert_any_call('homeassistant/binary_sensor/clouddetect_dev/safe/state', 'ON', retain=True)
    client.publish.reset_mock()
    ha.publish_discovery_configs()  # reconnect republishes the last data availability, not a blind online
    client.publish.assert_any_call('homeassistant/sensor/clouddetect_dev/data_availability', 'offline', retain=True)


def test_repeated_detection_failure_goes_unsafe(harness):
    make, advance, _ = harness
    monitor = make()
    advance(monitor, 120, new_hash=True)
    assert monitor.is_safe() is True

    monitor.cloud_detector.fail = True
    advance(monitor, 180)  # limit is max(180, 3 * 30) = 180
    assert monitor.is_safe() is True
    advance(monitor, 30)
    assert monitor.is_safe() is False



def test_recovery_after_detection_outage_needs_debounce(harness):
    make, advance, _ = harness
    monitor = make()
    advance(monitor, 120, new_hash=True)
    assert monitor._stable_safe_state is True

    monitor.cloud_detector.fail = True
    advance(monitor, 210)  # past the 180s limit
    assert monitor._stable_safe_state is False
    assert monitor.get_safety_history()[-1]['condition'] == 'Detection Stale'

    monitor.cloud_detector.fail = False
    advance(monitor, 30, new_hash=True)  # first clear detection starts the safe debounce
    assert monitor.is_safe() is False
    advance(monitor, 30, new_hash=True)
    assert monitor.is_safe() is False
    advance(monitor, 30, new_hash=True)  # 60s debounce_to_safe_sec elapsed
    assert monitor.is_safe() is True


def test_pending_safe_timer_does_not_survive_outage(harness):
    make, advance, _ = harness
    monitor = make()
    advance(monitor, 30, new_hash=True)  # one clear reading: safe transition pending
    assert monitor._pending_safe_state is True and monitor._stable_safe_state is False
    monitor.cloud_detector.fail = True
    advance(monitor, 600)
    monitor.cloud_detector.fail = False
    advance(monitor, 30, new_hash=True)  # first success after the outage
    assert monitor.is_safe() is False


def test_hung_thread_reads_unsafe_and_recovery_needs_debounce(harness):
    make, advance, clock = harness
    monitor = make()
    advance(monitor, 120, new_hash=True)
    assert monitor.is_safe() is True
    clock[0] += timedelta(seconds=600)  # detection thread hung: no cycles run
    assert monitor.is_safe() is False  # only the is_safe() read-side guard covers this
    advance(monitor, 30, new_hash=True)
    assert monitor.is_safe() is False
    assert monitor.get_safety_history()[-1]['condition'] == 'Detection Stale'


def test_legacy_mqtt_payload_has_no_image_hash(harness):
    make, advance, _ = harness
    monitor = make()
    published = []
    monitor.mqtt_client = types.SimpleNamespace(publish=lambda topic, payload: published.append(payload))
    monitor.detect_config.mqtt_discovery_mode = 'legacy'
    monitor.detect_config.topic = 'clouds'
    advance(monitor, 30, new_hash=True)
    assert published and 'image_hash' not in published[-1]
    assert json.loads(published[-1])['is_safe'] is False


def test_standalone_loop_pops_image_hash(monkeypatch):
    from detect import CloudDetector
    det = object.__new__(CloudDetector)
    det.detect = lambda: {'class_name': 'Clear', 'confidence_score': 99.0, 'image_hash': 'a'}
    published = []
    det.publish_result = published.append
    monkeypatch.setattr('detect.time.sleep', lambda s: (_ for _ in ()).throw(KeyboardInterrupt))
    with pytest.raises(KeyboardInterrupt):
        det.run_detection_loop()
    assert published == [{'class_name': 'Clear', 'confidence_score': 99.0}]


def test_load_image_hashes_raw_file_bytes(tmp_path):
    from PIL import Image
    from detect import CloudDetector
    path = tmp_path / 'sky.png'
    Image.new('RGB', (2, 2)).save(path)
    det = object.__new__(CloudDetector)
    _, hash1 = det._load_image(path.as_uri())
    _, hash2 = det._load_image(path.as_uri())
    assert hash1 == hash2 == hashlib.sha1(path.read_bytes()).hexdigest()
