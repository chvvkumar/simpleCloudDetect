import gpu_bootstrapper
gpu_bootstrapper.load_libs()
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- CONFIGURATION ---
BROKEN_MODEL_PATH = 'model.keras'
NEW_MODEL_PATH = 'model_clean.keras'
IMG_SIZE = (260, 260)
NUM_CLASSES = 7  # Based on your logs (Clear, Lightning, Meteors, Overcast, Partly Cloudy, Rain, Snow)

def rebuild_model():
    print("🔧 Rebuilding architecture from scratch (Clean Float32)...")
    
    # 1. Re-create the exact structure from train.py
    # Note: We skip Data Augmentation layers since we only need Inference now.
    
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights=None, # We will load weights later
        include_preprocessing=True 
    )
    base_model.trainable = False # Important to match structure

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    # Skip augmentation, go straight to base
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Re-create the head
    x = tf.keras.layers.Dense(NUM_CLASSES)(x) 
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

    clean_model = tf.keras.Model(inputs, outputs)
    return clean_model

def repair():
    # 1. Load the broken model (just to get weights)
    print(f"🚀 Loading weights from {BROKEN_MODEL_PATH}...")
    try:
        # We assume the file contains valid weights even if the config is messy
        # We load it blindly just to extract the weights
        old_model = tf.keras.models.load_model(BROKEN_MODEL_PATH, compile=False)
    except Exception as e:
        print(f"❌ Critical Load Error: {e}")
        return

    # 2. Build the new clean shell
    clean_model = rebuild_model()
    
    # 3. Transplant Weights
    # Since the structure (Base -> Head) is identical, we can blindly copy weights.
    # Note: If there's a mismatch due to the Augmentation layer in the old model, 
    # we might need to be specific. But usually 'load_weights' handles this.
    print("💉 Transplanting weights...")
    try:
        clean_model.set_weights(old_model.get_weights())
        print("✅ Weights transferred successfully.")
    except ValueError:
        print("⚠️ Direct weight transfer failed (Layer mismatch). Trying by name...")
        clean_model.load_weights(BROKEN_MODEL_PATH)

    # 4. Test on a Training Image (Sanity Check)
    train_img_path = "/mnt/f/MLClouds_incoming/resized/Clear/image_001.jpg" 
    
    if os.path.exists(train_img_path):
        print(f"\n🧪 Testing on known image: {train_img_path}")
        img = image.load_img(train_img_path, target_size=IMG_SIZE, interpolation='bicubic')
        img_arr = image.img_to_array(img)
        img_tensor = tf.expand_dims(img_arr, 0)
        
        preds = clean_model.predict(img_tensor, verbose=0)
        conf = 100 * np.max(preds[0])
        print(f"💪 New Confidence: {conf:.2f}% (Target: >90%)")
        
        if conf < 50:
            print("❌ Repair failed. Confidence is still low.")
            return

    # 5. Save the fixed model
    print(f"💾 Saving clean model to {NEW_MODEL_PATH}...")
    clean_model.save(NEW_MODEL_PATH)
    print("🎉 Done! You can now use this model for sorting.")

if __name__ == "__main__":
    repair()