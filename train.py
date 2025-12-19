import gpu_bootstrapper
gpu_bootstrapper.load_libs()
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import mixed_precision # Added for mixed precision
from pathlib import Path
import time

# --- Configuration ---
IMG_SIZE = (224, 224)       # Standard size for MobileNetV2
BATCH_SIZE = 32
EPOCHS_INITIAL = 10         # Epochs for frozen base training
EPOCHS_FINE = 10            # Epochs for fine-tuning
LEARNING_RATE = 0.0001
MODEL_SAVE_PATH = 'model.keras'
LABELS_SAVE_PATH = 'labels.txt'

def parse_args():
    parser = argparse.ArgumentParser(description='Train Cloud Detection Model')
    parser.add_argument('--data_dir', type=str, required=True, 
                        help='Path to the dataset directory containing class folders')
    return parser.parse_args()

def main():
    args = parse_args()
    data_dir = Path(args.data_dir)

    # 1. GPU & Mixed Precision Setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"✅ Found {len(gpus)} GPU(s). Training will be accelerated.")
        # Enable Mixed Precision (FP16)
        # This dramatically speeds up training on RTX cards and reduces VRAM usage
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"✅ Mixed Precision enabled: {policy.compute_dtype}")
    else:
        print("⚠️ No GPU found. Training will use CPU (slow).")

    # 2. Load Data
    if not data_dir.exists():
        print(f"❌ Error: Dataset directory '{data_dir}' not found.")
        return

    print(f"Loading dataset from: {data_dir}")
    try:
        train_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="training",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )

        val_ds = tf.keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="validation",
            seed=123,
            image_size=IMG_SIZE,
            batch_size=BATCH_SIZE
        )
    except ValueError as e:
        print(f"❌ Error loading dataset: {e}")
        print("Ensure the directory contains subfolders for each class.")
        return

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # 3. Optimize Data Loading
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. Data Augmentation
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.1),
    ])

    # 5. Create Model (MobileNetV2 Transfer Learning)
    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False  # Freeze base model

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # --- MODIFIED OUTPUT FOR MIXED PRECISION ---
    # We remove the activation from the Dense layer and add it separately.
    # The output activation MUST be float32 for numeric stability.
    x = tf.keras.layers.Dense(len(class_names))(x) 
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy']
    )

    # 6. Initial Training
    print("--- Starting Phase 1: Training Head ---")
    start_time = time.time()
    history = model.fit(
        train_ds,
        epochs=EPOCHS_INITIAL,
        validation_data=val_ds
    )
    print(f"Phase 1 completed in {time.time() - start_time:.2f} seconds.")

    # 7. Fine-Tuning
    print("--- Starting Phase 2: Fine-Tuning Base ---")
    base_model.trainable = True
    
    # Fine-tune from this layer onwards
    fine_tune_at = 100
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE/10),
        metrics=['accuracy']
    )
    
    total_epochs = EPOCHS_INITIAL + EPOCHS_FINE
    history_fine = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds
    )
    end_time = time.time()
    total_duration = end_time - start_time
    # Convert to minutes and seconds
    minutes = int(total_duration // 60)
    seconds = int(total_duration % 60)
    print(f"Phase 2 completed in {total_duration - (end_time - start_time):.2f} seconds.")
    print(f"⏱️ Total Training Time: {minutes}m {seconds}s")

    # 8. Save Artifacts
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)

    print(f"Saving labels to {LABELS_SAVE_PATH}...")
    with open(LABELS_SAVE_PATH, 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    print("✅ Training complete.")

if __name__ == "__main__":
    main()