"""
Regression test: MQTT availability must be restored after a reconnect.

The broker publishes the retained "offline" Last Will when a session drops.
paho's loop_start() reconnects silently, so without an on_connect handler that
republishes discovery, the device stays permanently unavailable in Home
Assistant while its state topics keep updating.
"""
import sys
import types
import unittest
from unittest import mock

# detect.py imports TensorFlow at module level. Stub it so this test runs
# without the ML stack installed. Everything else is a real dependency.
if 'tensorflow' not in sys.modules:
    keras_models = types.ModuleType('tensorflow.keras.models')
    keras_models.load_model = lambda *a, **k: None
    keras = types.ModuleType('tensorflow.keras')
    keras.models = keras_models
    tf = types.ModuleType('tensorflow')
    tf.keras = keras
    sys.modules['tensorflow'] = tf
    sys.modules['tensorflow.keras'] = keras
    sys.modules['tensorflow.keras.models'] = keras_models
    sys.modules['keras'] = keras
    sys.modules['keras.models'] = keras_models

sys.path.insert(0, '.')

from alpaca.device import AlpacaSafetyMonitor  # noqa: E402


class FakeConfig:
    broker = '192.168.1.250'
    port = 1883
    mqtt_username = None
    mqtt_password = None
    mqtt_discovery_mode = 'homeassistant'
    mqtt_discovery_prefix = 'homeassistant'
    device_id = 'testdevice'


class FakeMonitor:
    """Minimal stand-in so _setup_mqtt can run without loading the ML model."""

    def __init__(self):
        self.detect_config = FakeConfig()
        self.ha_discovery = None


class MqttReconnectTest(unittest.TestCase):

    def _setup(self):
        monitor = FakeMonitor()
        fake_client = mock.MagicMock()
        with mock.patch('alpaca.device.mqtt.Client', return_value=fake_client):
            AlpacaSafetyMonitor._setup_mqtt(monitor)
        return monitor, fake_client

    def test_last_will_is_offline_and_retained(self):
        _, client = self._setup()
        client.will_set.assert_called_once_with(
            'homeassistant/sensor/clouddetect_testdevice/availability',
            'offline',
            retain=True,
        )

    def test_reconnect_republishes_discovery(self):
        monitor, client = self._setup()
        monitor.ha_discovery = mock.MagicMock()

        client.on_connect(client, None, {}, 0)

        monitor.ha_discovery.publish_discovery_configs.assert_called_once()

    def test_failed_connect_does_not_republish(self):
        monitor, client = self._setup()
        monitor.ha_discovery = mock.MagicMock()

        client.on_connect(client, None, {}, 5)  # 5 = not authorised

        monitor.ha_discovery.publish_discovery_configs.assert_not_called()

    def test_first_connect_before_discovery_exists_does_not_raise(self):
        # on_connect fires from the network thread during _setup_mqtt, before
        # ha_discovery is assigned. It must not raise there.
        _, client = self._setup()
        client.on_connect(client, None, {}, 0)

    def test_paho2_signature_accepted(self):
        # paho 2.x calls on_connect with an extra properties argument.
        monitor, client = self._setup()
        monitor.ha_discovery = mock.MagicMock()

        client.on_connect(client, None, {}, 0, None)

        monitor.ha_discovery.publish_discovery_configs.assert_called_once()


if __name__ == '__main__':
    unittest.main()
