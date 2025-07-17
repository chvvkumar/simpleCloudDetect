#!/usr/bin/env python3

import logging
import platform
import os
from pathlib import Path
import h5py
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def detect_architecture():
    """Detect the current system architecture."""
    machine = platform.machine().lower()
    if machine in ('arm', 'armv7l', 'armv6l', 'aarch64'):
        return 'arm'
    return 'x86_64'  # Default to x86_64

def convert_h5_model(model_path: str = "keras_model.h5") -> None:
    """
    Convert Keras model by removing 'groups' attribute if present.
    
    Args:
        model_path: Path to the Keras model file
    """
    model_path = Path(model_path)
    if not model_path.exists():
        logger.error(f"Model file not found: {model_path}")
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        with h5py.File(model_path, mode="r+") as f:
            model_config = f.attrs.get("model_config")
            if model_config is None:
                logger.warning("No model_config found in model file")
                return

            if '"groups": 1,' in model_config:
                logger.info("Removing 'groups' attribute from model config")
                model_config = model_config.replace('"groups": 1,', '')
                f.attrs.modify('model_config', model_config)
                f.flush()
                
                # Verify changes
                updated_config = f.attrs.get("model_config")
                if '"groups": 1,' not in updated_config:
                    logger.info("Successfully removed 'groups' attribute")
                else:
                    logger.error("Failed to remove 'groups' attribute")
                    raise RuntimeError("Failed to remove 'groups' attribute")
            else:
                logger.info("No 'groups' attribute found in model config")

    except Exception as e:
        logger.error(f"Error converting model: {e}")
        raise

def convert_to_tflite(model_path: str = "keras_model.h5", output_path: str = "model.tflite") -> None:
    """
    Convert Keras model to TensorFlow Lite format for ARM architecture.
    
    Args:
        model_path: Path to the Keras model file
        output_path: Path where the TFLite model will be saved
    """
    try:
        # First, fix the model if needed
        convert_h5_model(model_path)
        
        # Import TensorFlow only when needed (not available on ARM)
        import tensorflow as tf
        
        # Load the Keras model
        logger.info(f"Loading Keras model from {model_path}")
        model = tf.keras.models.load_model(model_path, compile=False)
        
        # Convert the model to TFLite format
        logger.info("Converting model to TFLite format")
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        
        # Save the TFLite model
        with open(output_path, 'wb') as f:
            f.write(tflite_model)
        
        logger.info(f"TFLite model saved to {output_path}")
        
        # Return success status
        return True
    except Exception as e:
        logger.error(f"Error converting model to TFLite: {e}")
        raise

def convert_model():
    """
    Main conversion function that handles architecture-specific conversions.
    """
    arch = detect_architecture()
    logger.info(f"Detected architecture: {arch}")
    
    if arch == 'arm':
        try:
            # For ARM, convert to TFLite
            logger.info("ARM architecture detected, converting model to TFLite format")
            convert_to_tflite()
        except ImportError:
            # If TensorFlow is not available on ARM, try using the original model
            logger.warning("TensorFlow not available, falling back to standard conversion")
            convert_h5_model()
    else:
        # For x86, use the original conversion
        logger.info("x86 architecture detected, using standard conversion")
        convert_h5_model()

if __name__ == "__main__":
    try:
        convert_model()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
