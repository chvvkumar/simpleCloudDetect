import gpu_bootstrapper
gpu_bootstrapper.load_libs()
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- CONFIGURATION ---
BROKEN_MODEL_PATH = 'model.keras'
NEW_MODEL_PATH = 'model_clean.keras'
IMG_SIZE = (260, 260)  # EfficientNetV2-S Input Size
NUM_CLASSES = 7        # Your 7 cloud classes

def rebuild_architecture():
    print("🔧 Rebuilding EfficientNetV2S architecture (Float32)...")
    
    # 1. Base Model (EfficientNetV2S)
    # include_preprocessing=True handles the 0-255 to internal scaling automatically
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None, # We will load weights from file
        include_preprocessing=True 
    )
    base_model.trainable = False

    # 2. Re-create the Head
    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # We skip Data Augmentation layers for the clean inference model
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    x = tf.keras.layers.Dense(NUM_CLASSES)(x) 
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

    clean_model = tf.keras.Model(inputs, outputs)
    return clean_model

def repair():
    print(f"🚀 Loading weights from {BROKEN_MODEL_PATH}...")
    try:
        # compile=False helps avoid optimizer config errors
        old_model = tf.keras.models.load_model(BROKEN_MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ Load Error: {e}")
        return

    # Build clean shell
    clean_model = rebuild_architecture()
    
    print("💉 Transplanting weights...")
    try:
        # Try direct transfer first
        clean_model.set_weights(old_model.get_weights())
        print("✅ Weights transferred successfully (Direct Match).")
    except Exception:
        print("⚠️ Direct transfer failed (Layer mismatch). Attempting smart transfer...")
        
        # Smart Transfer: Copy Base and Head separately
        # 1. Base
        try:
            old_base = old_model.get_layer('efficientnetv2-s')
            new_base = clean_model.get_layer('efficientnetv2-s')
            new_base.set_weights(old_base.get_weights())
            print("   - Base model weights transferred.")
        except:
            print("   ❌ Could not find/match 'efficientnetv2-s' layer.")
            
        # 2. Dense Head
        try:
            # Find the Dense layer in both
            old_dense = [l for l in old_model.layers if isinstance(l, tf.keras.layers.Dense)][-1]
            new_dense = [l for l in clean_model.layers if isinstance(l, tf.keras.layers.Dense)][-1]
            new_dense.set_weights(old_dense.get_weights())
            print("   - Dense layer weights transferred.")
        except:
             print("   ❌ Could not find/match Dense layer.")

    # Save
    print(f"💾 Saving clean model to {NEW_MODEL_PATH}...")
    clean_model.save(NEW_MODEL_PATH)
    print("🎉 Done! Please update sort_bulk.py to use 'model_clean.keras'")

if __name__ == "__main__":
    repair()