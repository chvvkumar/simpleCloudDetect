import os
import unittest
from unittest.mock import patch

from alpaca.config import AlpacaConfig


class TestResolveDetectOverrides(unittest.TestCase):

    def _config(self, **overrides):
        """Build an AlpacaConfig with the given UI field overrides.

        Construction reads env via default_factory, so callers patch the
        environment before instantiating.
        """
        return AlpacaConfig(**overrides)

    def test_blank_broker_inherits_env(self):
        """Blank UI broker inherits the MQTT_BROKER env value."""
        with patch.dict(os.environ, {'MQTT_BROKER': 'env-broker.example.com'}):
            cfg = self._config(mqtt_broker='')
            self.assertEqual(
                cfg.resolve_detect_overrides()['broker'],
                'env-broker.example.com',
            )

    def test_filled_broker_overrides_env(self):
        """A filled UI broker wins over the env value."""
        with patch.dict(os.environ, {'MQTT_BROKER': 'env-broker.example.com'}):
            cfg = self._config(mqtt_broker='ui-broker.example.com')
            self.assertEqual(
                cfg.resolve_detect_overrides()['broker'],
                'ui-broker.example.com',
            )

    def test_blank_image_url_inherits_env(self):
        """Blank UI image_url inherits IMAGE_URL env value."""
        with patch.dict(os.environ, {'IMAGE_URL': 'http://env/image.jpg'}):
            cfg = self._config(image_url='')
            self.assertEqual(
                cfg.resolve_detect_overrides()['image_url'],
                'http://env/image.jpg',
            )

    def test_filled_image_url_overrides_env(self):
        """A filled UI image_url wins over the env value."""
        with patch.dict(os.environ, {'IMAGE_URL': 'http://env/image.jpg'}):
            cfg = self._config(image_url='http://ui/image.jpg')
            self.assertEqual(
                cfg.resolve_detect_overrides()['image_url'],
                'http://ui/image.jpg',
            )

    def test_port_filled_int_wins(self):
        """A filled int mqtt_port wins over env."""
        with patch.dict(os.environ, {'MQTT_PORT': '8883'}):
            cfg = self._config(mqtt_port=1234)
            result = cfg.resolve_detect_overrides()
            self.assertEqual(result['port'], 1234)
            self.assertIsInstance(result['port'], int)

    def test_port_falls_back_to_env(self):
        """Empty/zero mqtt_port falls back to the MQTT_PORT env value as int."""
        with patch.dict(os.environ, {'MQTT_PORT': '8883'}):
            cfg = self._config(mqtt_port=0)
            result = cfg.resolve_detect_overrides()
            self.assertEqual(result['port'], 8883)
            self.assertIsInstance(result['port'], int)

    def test_port_defaults_to_1883(self):
        """With no UI value and no env, port defaults to 1883."""
        with patch.dict(os.environ, {'MQTT_PORT': ''}):
            cfg = self._config(mqtt_port=0)
            self.assertEqual(cfg.resolve_detect_overrides()['port'], 1883)

    def test_verify_ssl_true_passes_through(self):
        """verify_ssl True passes through as bool True."""
        with patch.dict(os.environ, {}):
            cfg = self._config(verify_ssl=True)
            result = cfg.resolve_detect_overrides()
            self.assertIs(result['verify_ssl'], True)

    def test_verify_ssl_false_passes_through(self):
        """verify_ssl False passes through as bool False."""
        with patch.dict(os.environ, {}):
            cfg = self._config(verify_ssl=False)
            result = cfg.resolve_detect_overrides()
            self.assertIs(result['verify_ssl'], False)

    def test_discovery_mode_lowercased(self):
        """A filled mqtt_discovery_mode is lowercased."""
        with patch.dict(os.environ, {'MQTT_DISCOVERY_MODE': 'legacy'}):
            cfg = self._config(mqtt_discovery_mode='HomeAssistant')
            self.assertEqual(
                cfg.resolve_detect_overrides()['mqtt_discovery_mode'],
                'homeassistant',
            )

    def test_discovery_mode_falls_back_to_env(self):
        """Empty mqtt_discovery_mode falls back to env (also lowercased)."""
        with patch.dict(os.environ, {'MQTT_DISCOVERY_MODE': 'HomeAssistant'}):
            cfg = self._config(mqtt_discovery_mode='')
            self.assertEqual(
                cfg.resolve_detect_overrides()['mqtt_discovery_mode'],
                'homeassistant',
            )

    def test_broker_blank_resolves_to_none(self):
        """Blank broker with no env resolves to None (MQTT disabled)."""
        with patch.dict(os.environ, {'MQTT_BROKER': ''}):
            cfg = self._config(mqtt_broker='')
            self.assertIsNone(cfg.resolve_detect_overrides()['broker'])

    def test_username_password_blank_resolve_to_none(self):
        """Blank username/password with no env resolve to None."""
        with patch.dict(os.environ, {'MQTT_USERNAME': '', 'MQTT_PASSWORD': ''}):
            cfg = self._config(mqtt_username='', mqtt_password='')
            result = cfg.resolve_detect_overrides()
            self.assertIsNone(result['mqtt_username'])
            self.assertIsNone(result['mqtt_password'])

    def test_discovery_prefix_default(self):
        """Blank prefix with the env var absent defaults to homeassistant."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop('MQTT_DISCOVERY_PREFIX', None)
            cfg = self._config(mqtt_discovery_prefix='')
            self.assertEqual(
                cfg.resolve_detect_overrides()['mqtt_discovery_prefix'],
                'homeassistant',
            )

    def test_discovery_prefix_filled_overrides_env(self):
        """A filled prefix wins over the env value."""
        with patch.dict(os.environ, {'MQTT_DISCOVERY_PREFIX': 'env-prefix'}):
            cfg = self._config(mqtt_discovery_prefix='ui-prefix')
            self.assertEqual(
                cfg.resolve_detect_overrides()['mqtt_discovery_prefix'],
                'ui-prefix',
            )

    def test_returned_keys_match_contract(self):
        """The override dict exposes exactly the contracted detect keys."""
        with patch.dict(os.environ, {}):
            cfg = self._config()
            keys = set(cfg.resolve_detect_overrides().keys())
            self.assertEqual(
                keys,
                {
                    'image_url', 'broker', 'port', 'mqtt_username',
                    'mqtt_password', 'mqtt_discovery_mode',
                    'mqtt_discovery_prefix', 'device_id', 'device_name',
                    'verify_ssl',
                },
            )


if __name__ == '__main__':
    unittest.main()
