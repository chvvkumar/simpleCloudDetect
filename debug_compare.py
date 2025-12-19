import gpu_bootstrapper
gpu_bootstrapper.load_libs()
import tensorflow as tf
# REMOVED: from tensorflow.keras import mixed_precision 
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- CRITICAL CHANGE: NO MIXED PRECISION POLICY ---
# We let the model load in standard 32-bit mode.
# This prevents the "30% confidence" bug.
print("🔧 Policy: Default (Float32)")

MODEL_PATH = 'model.keras'
IMG_SIZE = (260, 260)

def test_file(filepath, label):
    print(f"\n--- Testing {label} ---")
    print(f"📄 File: {filepath}")
    
    if not os.path.exists(filepath):
        print("❌ File not found!")
        return

    # Load exactly as we do in the sorter
    img = image.load_img(filepath, target_size=IMG_SIZE, interpolation='bicubic')
    img_arr = image.img_to_array(img)
    img_tensor = tf.expand_dims(img_arr, 0) 

    # Load Model (Lazy load)
    global model
    if 'model' not in globals():
        print("🚀 Loading Model...")
        # compile=False makes loading faster and avoids optimizer variable warnings
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # Predict
    preds = model.predict(img_tensor, verbose=0)
    score = tf.nn.softmax(preds[0])
    confidence = 100 * np.max(score)
    top_class_idx = np.argmax(score)
    
    print(f"🎯 Prediction: Class Index {top_class_idx}")
    print(f"💪 Confidence: {confidence:.2f}%")

# Using the paths you confirmed exist
train_file = "/mnt/f/MLClouds_incoming/resized/Clear/image_001.jpg" 
sort_file = "/mnt/f/delete/source/image_001.jpg"

test_file(train_file, "TRAINING IMAGE")
test_file(sort_file, "SORTING IMAGE")