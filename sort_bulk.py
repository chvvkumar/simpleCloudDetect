import gpu_bootstrapper
gpu_bootstrapper.load_libs()
import os
import shutil
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import math
import time

# --- Configuration ---
MODEL_PATH = 'model.keras'
LABELS_PATH = 'labels.txt'
IMG_SIZE = (260, 260) 
BATCH_SIZE = 64  

def load_labels():
    with open(LABELS_PATH, 'r') as f:
        return [line.strip() for line in f.readlines()]

def create_class_folders(base_dir, class_names):
    paths = {}
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
        
    for name in class_names:
        folder_path = os.path.join(base_dir, name)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        paths[name] = folder_path
    
    uncert_path = os.path.join(base_dir, "_Uncertain")
    if not os.path.exists(uncert_path):
        os.makedirs(uncert_path)
    paths["_Uncertain"] = uncert_path
    
    return paths

def get_unique_path(dest_folder, filename):
    """
    If file exists in destination, append a counter: image.jpg -> image_1.jpg
    """
    base, ext = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    full_path = os.path.join(dest_folder, new_filename)
    
    while os.path.exists(full_path):
        new_filename = f"{base}_{counter}{ext}"
        full_path = os.path.join(dest_folder, new_filename)
        counter += 1
        
    return full_path

def sort_images(source_dir, output_dir, confidence_threshold=70.0):
    print(f"🚀 Loading Model from {MODEL_PATH}...")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return

    class_names = load_labels()
    dest_folders = create_class_folders(output_dir, class_names)
    
    # --- MODIFIED: RECURSIVE SCAN ---
    print(f"📂 Scanning {source_dir} recursively...")
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
    all_files_full_paths = []
    
    # os.walk automatically dives into all subfolders (e.g. 20241230, 20241231)
    for root, dirs, files in os.walk(source_dir):
        for f in files:
            if f.lower().endswith(valid_exts):
                # Store the FULL path so we can locate it later
                all_files_full_paths.append(os.path.join(root, f))
    
    total_files = len(all_files_full_paths)
    print(f"📂 Found {total_files} images in total.")
    print(f"⚡ Processing in batches of {BATCH_SIZE}...")
    print("-" * 50)

    num_batches = math.ceil(total_files / BATCH_SIZE)
    moved_count = 0
    start_time = time.time()

    for b in range(num_batches):
        batch_paths = all_files_full_paths[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
        batch_images = []
        valid_batch_paths = []

        # Load images
        for src_path in batch_paths:
            try:
                img = image.load_img(src_path, target_size=IMG_SIZE)
                img_arr = image.img_to_array(img)
                batch_images.append(img_arr)
                valid_batch_paths.append(src_path)
            except Exception as e:
                print(f"⚠️ Corrupt file skipped: {src_path}")

        if not batch_images:
            continue

        # Predict
        batch_arr = np.array(batch_images)
        predictions = model.predict_on_batch(batch_arr)

        # Move Files
        for i, src_path in enumerate(valid_batch_paths):
            scores = tf.nn.softmax(predictions[i])
            pred_idx = np.argmax(scores)
            confidence = 100 * np.max(scores)
            
            # Use original filename
            filename = os.path.basename(src_path)
            
            if confidence >= confidence_threshold:
                target_folder = dest_folders[class_names[pred_idx]]
            else:
                target_folder = dest_folders["_Uncertain"]
            
            # Generate unique destination path to prevent overwrites
            dst_path = get_unique_path(target_folder, filename)
            
            try:
                shutil.move(src_path, dst_path)
                moved_count += 1
            except Exception as e:
                print(f"❌ Error moving {filename}: {e}")

        # Progress
        if b % 5 == 0:
            print(f"   Batch {b+1}/{num_batches} complete. ({moved_count} moved)")

    duration = (time.time() - start_time) / 60
    print("-" * 50)
    print(f"✅ Sorting Complete!")
    print(f"⏱️ Time: {duration:.2f} minutes")
    print(f"📦 Total Images Sorted: {moved_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, required=True, help='Root folder containing dated subfolders')
    parser.add_argument('--output', type=str, required=True, help='Root folder to create class subfolders in')
    parser.add_argument('--conf', type=float, default=60.0, help='Confidence threshold')
    args = parser.parse_args()
    
    sort_images(args.source, args.output, args.conf)