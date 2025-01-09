#!/usr/bin/env python3

import logging
from pathlib import Path
import h5py

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def convert_model(model_path: str = "keras_model.h5") -> None:
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

if __name__ == "__main__":
    try:
        convert_model()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        raise
