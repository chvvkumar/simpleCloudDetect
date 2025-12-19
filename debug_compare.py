import gpu_bootstrapper
gpu_bootstrapper.load_libs()
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- 1. Match Training Environment EXACTLY ---
# We enable mixed precision because the model was trained with it.
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
print(f"🔧 Policy set to: {policy.compute_dtype}")

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
    img_tensor = tf.expand_dims(img_arr, 0) # Shape: (1, 260, 260, 3)

    print(f"📊 Input Stats: Min={img_arr.min():.1f}, Max={img_arr.max():.1f}, Shape={img_arr.shape}")
    
    # Load Model (Lazy load)
    global model
    if 'model' not in globals():
        print("🚀 Loading Model...")
        model = tf.keras.models.load_model(MODEL_PATH)
    
    # Predict
    preds = model.predict(img_tensor, verbose=0)
    score = tf.nn.softmax(preds[0])
    confidence = 100 * np.max(score)
    top_class_idx = np.argmax(score)
    
    # Use generic index if labels missing
    print(f"🎯 Prediction: Class Index {top_class_idx}")
    print(f"💪 Confidence: \033[92m{confidence:.2f}%\033[0m")

# --- USER: UPDATE THESE TWO PATHS ---
# 1. Pick a file that you KNOW is in the training set (from /resized/)
train_file = "/mnt/f/MLClouds_incoming/resized/Clear/image_001.jpg" 

# 2. Pick a file from the folder that is failing (from /source/)
sort_file = "/mnt/f/delete/source/image-20250104042554.jpg"

# ------------------------------------
# Try to find a real file if the hardcoded ones don't exist
if not os.path.exists(train_file):
    # Auto-find a file in training dir
    tr_dir = "/mnt/f/MLClouds_incoming/resized/"
    for root, _, files in os.walk(tr_dir):
        if files:
            train_file = os.path.join(root, files[0])
            break

if not os.path.exists(sort_file):
    # Auto-find a file in source dir
    src_dir = "/mnt/f/delete/source/"
    for root, _, files in os.walk(src_dir):
        if files:
            sort_file = os.path.join(root, files[0])
            break

test_file(train_file, "TRAINING IMAGE (Should be ~95%)")
test_file(sort_file, "SORTING IMAGE (Currently ~30%)")