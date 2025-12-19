import gpu_bootstrapper
gpu_bootstrapper.load_libs()
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision
from pathlib import Path
import time
import subprocess
import datetime

# --- Configuration ---
IMG_SIZE = (260, 260)       
BATCH_SIZE = 64             
EPOCHS_INITIAL = 20         # Increased, because EarlyStopping will cut it short if needed
EPOCHS_FINE = 20            # Increased cap
LEARNING_RATE = 1e-3
MODEL_SAVE_PATH = 'model.keras'
LABELS_SAVE_PATH = 'labels.txt'

# --- Custom GPU Monitor ---
class GPULogger(tf.keras.callbacks.Callback):
    def __init__(self, log_freq=50):
        self.log_freq = log_freq

    def on_train_batch_end(self, batch, logs=None):
        if batch % self.log_freq == 0 and batch > 0:
            try:
                cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw,temperature.gpu --format=csv,noheader,nounits"
                result = subprocess.check_output(cmd, shell=True).decode('utf-8').strip()
                util, mem, power, temp = result.split(',')
                now = datetime.datetime.now().strftime("%H:%M:%S")
                util = float(util)
                color = "\033[92m" if util > 80 else "\033[91m"
                reset = "\033[0m"
                print(f"\n   [GPU {now}] Util: {color}{util}%{reset} | Mem: {mem.strip()}MB | Pwr: {power.strip()}W | Temp: {temp.strip()}C")
            except Exception:
                pass

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cloud Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to dataset')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # 1. Setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s).")
        mixed_precision.set_global_policy('mixed_float16')
    else:
        print("⚠️ No GPU found.")

    # 2. Data
    if not data_dir.exists():
        print(f"❌ Error: {data_dir} not found.")
        return

    print(f"Loading dataset from: {data_dir}")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="training", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir, validation_split=0.2, subset="validation", seed=123,
        image_size=IMG_SIZE, batch_size=BATCH_SIZE
    )
    class_names = train_ds.class_names

    # Optimize (No .cache() to prevent OOM)
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

    # 3. Model
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        tf.keras.layers.RandomContrast(0.2),
    ])

    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet', include_preprocessing=True
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation='softmax', dtype='float32')(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # --- DEFINE SAFETY CALLBACKS ---
    callbacks_list = [
        # 1. GPU Monitor
        GPULogger(log_freq=50),
        
        # 2. Save ONLY the best version (Prevents saving a "worse" model at the end)
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),

        # 3. Stop if stuck for 4 epochs (Prevents wasting time)
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=4,
            restore_best_weights=True,
            verbose=1
        )
    ]

    # 4. Phase 1 Training
    print("--- Starting Phase 1: Training Head ---")
    t0 = time.time()
    history = model.fit(
        train_ds, 
        epochs=EPOCHS_INITIAL, 
        validation_data=val_ds,
        callbacks=callbacks_list  # <--- Safety nets active
    )
    t1 = time.time()
    print(f"✅ Phase 1 Duration: {(t1 - t0)/60:.2f} minutes")

    # 5. Phase 2 Training
    print("--- Starting Phase 2: Fine-Tuning Base ---")
    base_model.trainable = True
    for layer in base_model.layers[:-100]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE/10),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # We must RE-CREATE callbacks for Phase 2 to reset their internal counters
    callbacks_list_phase2 = [
        GPULogger(log_freq=50),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH,
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5, # Give it slightly more patience during fine-tuning
            restore_best_weights=True,
            verbose=1
        )
    ]

    t2 = time.time()
    history_fine = model.fit(
        train_ds,
        epochs=EPOCHS_INITIAL + EPOCHS_FINE,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds,
        callbacks=callbacks_list_phase2
    )
    t3 = time.time()
    print(f"✅ Phase 2 Duration: {(t3 - t2)/60:.2f} minutes")
    print(f"⏱️ Total Training Time: {(t3 - t0)/60:.2f} minutes")

    # Save labels
    print(f"Saving labels to {LABELS_SAVE_PATH}...")
    with open(LABELS_SAVE_PATH, 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    print("✅ Training complete (Best model saved automatically).")

if __name__ == "__main__":
    main()