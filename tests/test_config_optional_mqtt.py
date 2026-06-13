import os
import unittest
from unittest.mock import patch
from detect import Config

class TestConfigOptionalMqtt(unittest.TestCase):

    def setUp(self):
        # Clear relevant environment variables before each test
        self.env_vars = {
            'IMAGE_URL': 'http://example.com/image.jpg',
            'DETECT_INTERVAL': '60',
            'MQTT_BROKER': '',
            'MQTT_PORT': '',
            'MQTT_TOPIC': '',
            'MQTT_USERNAME': '',
            'MQTT_PASSWORD': '',
            'MQTT_DISCOVERY_MODE': '',
            'DEVICE_ID': ''
        }
        self.patcher = patch.dict(os.environ, self.env_vars)
        self.patcher.start()

    def tearDown(self):
        self.patcher.stop()

    def test_config_no_mqtt(self):
        """Test that Config can be initialized without MQTT settings."""
        # Ensure MQTT variables are unset/empty
        if 'MQTT_BROKER' in os.environ:
            del os.environ['MQTT_BROKER']
            
        config = Config.from_env()
        
        self.assertIsNone(config.broker)
        self.assertIsNone(config.topic)
        self.assertEqual(config.image_url, 'http://example.com/image.jpg')

    def test_config_with_mqtt(self):
        """Test that Config is initialized correctly with MQTT settings."""
        os.environ['MQTT_BROKER'] = 'mqtt.example.com'
        os.environ['MQTT_TOPIC'] = 'test/topic'
        
        config = Config.from_env()
        
        self.assertEqual(config.broker, 'mqtt.example.com')
        self.assertEqual(config.topic, 'test/topic')

    def test_config_legacy_mode_missing_topic(self):
        """Test that legacy mode requires topic only if broker is set."""
        os.environ['MQTT_BROKER'] = 'mqtt.example.com'
        os.environ['MQTT_DISCOVERY_MODE'] = 'legacy'
        # MQTT_TOPIC is missing
        
        with self.assertRaises(ValueError) as cm:
            Config.from_env()
        self.assertIn("Missing required environment variables", str(cm.exception))

    def test_config_ha_mode_missing_device_id(self):
        """Test that HA mode requires DEVICE_ID only if broker is set."""
        os.environ['MQTT_BROKER'] = 'mqtt.example.com'
        os.environ['MQTT_DISCOVERY_MODE'] = 'homeassistant'
        # DEVICE_ID is missing
        
        with self.assertRaises(ValueError) as cm:
            Config.from_env()
        self.assertIn("DEVICE_ID is required", str(cm.exception))

if __name__ == '__main__':
    unittest.main()
