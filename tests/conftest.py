"""Let route-level unit tests import the package without TensorFlow installed."""
import sys
import types

try:
    import keras.models  # noqa: F401
except ImportError:
    keras = types.ModuleType("keras")
    keras.models = types.ModuleType("keras.models")
    keras.models.load_model = lambda *a, **k: None
    sys.modules["keras"] = keras
    sys.modules["keras.models"] = keras.models
