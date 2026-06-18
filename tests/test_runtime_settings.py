"""Unit tests for AlpacaSafetyMonitor.apply_runtime_settings() (live reload)."""
import types
import unittest
from unittest.mock import MagicMock, patch

from alpaca.device import AlpacaSafetyMonitor


def _make_detect_config():
    """A minimal stand-in for detect.Config with the attributes the method touches."""
    cfg = types.SimpleNamespace()
    cfg.image_url = 'http://old/image.jpg'
    cfg.broker = 'old-broker'
    cfg.port = 1883
    cfg.topic = 'old/topic'
    cfg.mqtt_username = None
    cfg.mqtt_password = None
    cfg.mqtt_discovery_mode = 'legacy'
    cfg.mqtt_discovery_prefix = 'homeassistant'
    cfg.device_id = 'old-id'
    cfg.device_name = 'Old Device'
    cfg.verify_ssl = False
    return cfg


def _make_monitor(overrides, detect_config=None):
    """Build a bare AlpacaSafetyMonitor without running __init__ (no ML model load).

    Only the attributes apply_runtime_settings() reads/writes are populated.
    """
    monitor = AlpacaSafetyMonitor.__new__(AlpacaSafetyMonitor)
    monitor.detect_config = detect_config or _make_detect_config()
    monitor.alpaca_config = MagicMock()
    monitor.alpaca_config.resolve_detect_overrides.return_value = overrides
    monitor.mqtt_client = MagicMock()
    monitor.cloud_detector = MagicMock()
    monitor.ha_discovery = MagicMock()
    return monitor


def _overrides(**kw):
    base = {
        'image_url': 'http://new/image.jpg',
        'broker': 'new-broker',
        'port': 8883,
        'mqtt_username': 'user',
        'mqtt_password': 'pass',
        'mqtt_discovery_mode': 'legacy',
        'mqtt_discovery_prefix': 'homeassistant',
        'device_id': 'new-id',
        'device_name': 'New Device',
        'verify_ssl': True,
    }
    base.update(kw)
    return base


class TestApplyRuntimeSettings(unittest.TestCase):

    def test_detect_config_mutated_from_overrides(self):
        """Each override key is written onto detect_config in place."""
        monitor = _make_monitor(_overrides())
        new_client = MagicMock()
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=new_client):
            monitor.apply_runtime_settings()

        cfg = monitor.detect_config
        self.assertEqual(cfg.image_url, 'http://new/image.jpg')
        self.assertEqual(cfg.broker, 'new-broker')
        self.assertEqual(cfg.port, 8883)
        self.assertEqual(cfg.mqtt_username, 'user')
        self.assertEqual(cfg.mqtt_password, 'pass')
        self.assertEqual(cfg.mqtt_discovery_mode, 'legacy')
        self.assertEqual(cfg.mqtt_discovery_prefix, 'homeassistant')
        self.assertEqual(cfg.device_id, 'new-id')
        self.assertEqual(cfg.device_name, 'New Device')
        self.assertIs(cfg.verify_ssl, True)

    def test_image_url_updated_for_next_detect(self):
        """image_url on detect_config is updated from resolve_detect_overrides."""
        monitor = _make_monitor(_overrides(image_url='http://changed/img.jpg'))
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=MagicMock()):
            monitor.apply_runtime_settings()
        self.assertEqual(monitor.detect_config.image_url, 'http://changed/img.jpg')

    def test_mqtt_client_rebuilt_and_propagated(self):
        """_setup_mqtt is called and the new client reaches monitor and detector."""
        monitor = _make_monitor(_overrides())
        new_client = MagicMock(name='new_client')
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=new_client) as m:
            monitor.apply_runtime_settings()
        m.assert_called_once()
        self.assertIs(monitor.mqtt_client, new_client)
        self.assertIs(monitor.cloud_detector.mqtt_client, new_client)

    def test_old_homeassistant_client_torn_down(self):
        """An existing HA client gets offline availability published then disconnected."""
        cfg = _make_detect_config()
        cfg.mqtt_discovery_mode = 'homeassistant'
        cfg.mqtt_discovery_prefix = 'homeassistant'
        cfg.device_id = 'old-id'
        monitor = _make_monitor(_overrides(mqtt_discovery_mode='legacy'), detect_config=cfg)
        old_client = monitor.mqtt_client
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=MagicMock()):
            monitor.apply_runtime_settings()

        # Offline availability published to the old device's availability topic.
        published = [c.args for c in old_client.publish.call_args_list]
        self.assertTrue(
            any('availability' in str(args[0]) and args[1] == 'offline' for args in published),
            f"expected an offline availability publish, got {published}",
        )
        old_client.loop_stop.assert_called_once()
        old_client.disconnect.assert_called_once()

    def test_ha_discovery_reinit_when_homeassistant(self):
        """Switching to HA mode builds a fresh HADiscoveryManager and publishes configs."""
        monitor = _make_monitor(_overrides(mqtt_discovery_mode='homeassistant'))
        new_client = MagicMock()
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=new_client), \
             patch('alpaca.device.HADiscoveryManager') as ha_cls:
            monitor.apply_runtime_settings()
        ha_cls.assert_called_once_with(monitor.detect_config, new_client)
        ha_cls.return_value.publish_discovery_configs.assert_called_once()
        self.assertIs(monitor.ha_discovery, ha_cls.return_value)

    def test_ha_discovery_cleared_in_legacy_mode(self):
        """Legacy mode leaves ha_discovery as None."""
        monitor = _make_monitor(_overrides(mqtt_discovery_mode='legacy'))
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=MagicMock()), \
             patch('alpaca.device.HADiscoveryManager') as ha_cls:
            monitor.apply_runtime_settings()
        ha_cls.assert_not_called()
        self.assertIsNone(monitor.ha_discovery)

    def test_ha_discovery_none_when_no_client(self):
        """HA mode with no MQTT client (broker unset) leaves ha_discovery None."""
        monitor = _make_monitor(_overrides(mqtt_discovery_mode='homeassistant', broker=None))
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=None), \
             patch('alpaca.device.HADiscoveryManager') as ha_cls:
            monitor.apply_runtime_settings()
        ha_cls.assert_not_called()
        self.assertIsNone(monitor.ha_discovery)
        self.assertIsNone(monitor.mqtt_client)

    def test_setup_mqtt_returning_none_does_not_raise(self):
        """A broker that fails to connect (None client) must not propagate."""
        monitor = _make_monitor(_overrides())
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=None):
            try:
                monitor.apply_runtime_settings()
            except Exception as e:  # noqa: BLE001
                self.fail(f"apply_runtime_settings raised unexpectedly: {e}")
        self.assertIsNone(monitor.mqtt_client)
        self.assertIsNone(monitor.cloud_detector.mqtt_client)

    def test_setup_mqtt_raising_does_not_propagate(self):
        """A failure inside the rebuild path is swallowed; the method never raises."""
        monitor = _make_monitor(_overrides())
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', side_effect=RuntimeError('boom')):
            try:
                monitor.apply_runtime_settings()
            except Exception as e:  # noqa: BLE001
                self.fail(f"apply_runtime_settings raised unexpectedly: {e}")

    def test_teardown_failure_does_not_propagate(self):
        """A failure while tearing down the old client is swallowed."""
        cfg = _make_detect_config()
        cfg.mqtt_discovery_mode = 'homeassistant'
        monitor = _make_monitor(_overrides(), detect_config=cfg)
        monitor.mqtt_client.disconnect.side_effect = RuntimeError('disconnect failed')
        with patch.object(AlpacaSafetyMonitor, '_setup_mqtt', return_value=MagicMock()):
            try:
                monitor.apply_runtime_settings()
            except Exception as e:  # noqa: BLE001
                self.fail(f"apply_runtime_settings raised unexpectedly: {e}")


if __name__ == '__main__':
    unittest.main()
