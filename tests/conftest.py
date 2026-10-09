"""Shared test setup: repo root on sys.path and a Keras stub so detect.py imports without the ML stack."""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import keras.models  # noqa: F401
except ImportError:
    keras_models = types.ModuleType('keras.models')
    keras_models.load_model = lambda *a, **k: None
    keras = types.ModuleType('keras')
    keras.models = keras_models
    tf = types.ModuleType('tensorflow')
    tf.keras = keras
    sys.modules['tensorflow'] = tf
    sys.modules['tensorflow.keras'] = keras
    sys.modules['tensorflow.keras.models'] = keras_models
    sys.modules['keras'] = keras
    sys.modules['keras.models'] = keras_models
