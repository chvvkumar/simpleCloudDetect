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

# --- Configuration ---
# EfficientNetV2S works best at slightly higher resolutions, but 260 is a good balance.
IMG_SIZE = (260, 260)       
# RTX 5070 Ti can easily handle 128 with Mixed Precision.
# If you get OOM (Out of Memory) errors with 3x data, drop this to 64.
BATCH_SIZE = 128            
EPOCHS_INITIAL = 10
EPOCHS_FINE = 10
LEARNING_RATE = 1e-3        # Slightly higher initial rate for larger batch size
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
        print(f"✅ Found {len(gpus)} GPU(s).")
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
        return

    class_names = train_ds.class_names
    print(f"Classes found: {class_names}")

    # 3. Optimize Data Loading
    AUTOTUNE = tf.data.AUTOTUNE
    # Note: With 60k+ images, .cache() might fill your RAM. 
    # If your PC slows down, remove .cache() and just keep .prefetch()
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # 4. Data Augmentation (Enhanced for Clouds)
    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip('horizontal'),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomZoom(0.2),
        # Clouds depend heavily on lighting; this helps the model generalize
        tf.keras.layers.RandomContrast(0.2), 
    ])

    # 5. Create Model (EfficientNetV2S)
    # Note: EfficientNetV2 expects [0-255] inputs, so we DO NOT manually preprocess/scale.
    
    base_model = tf.keras.applications.EfficientNetV2S(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights='imagenet',
        include_preprocessing=True # Internal rescaling logic
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = data_augmentation(inputs)
    # We feed raw inputs directly; the model handles normalization
    x = base_model(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    
    # Output Layer
    x = tf.keras.layers.Dense(len(class_names))(x) 
    outputs = tf.keras.layers.Activation('softmax', dtype='float32', name='predictions')(x)

    model = tf.keras.Model(inputs, outputs)

    # COMPILE WITH XLA (jit_compile=True)
    # This fuses operations for the RTX 5070 Ti
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        metrics=['accuracy'],
        jit_compile=True 
    )

    # 6. Initial Training
    print("--- Starting Phase 1: Training Head ---")
    t0 = time.time()
    history = model.fit(train_ds, epochs=EPOCHS_INITIAL, validation_data=val_ds)
    t1 = time.time()
    print(f"✅ Phase 1 Duration: {(t1 - t0)/60:.2f} minutes")

    # 7. Fine-Tuning
    print("--- Starting Phase 2: Fine-Tuning Base ---")
    base_model.trainable = True
    
    total_epochs = EPOCHS_INITIAL + EPOCHS_FINE
    
    # Unfreeze the top layers. EfficientNetV2S is deep; unfreezing the top 50-100 is usually enough.
    # We calculate the starting point dynamically.
    for layer in base_model.layers[:-100]:
        layer.trainable = False

    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
        optimizer=tf.keras.optimizers.RMSprop(learning_rate=LEARNING_RATE/10),
        metrics=['accuracy'],
        jit_compile=True # Keep XLA on
    )
    
    t2 = time.time()
    history_fine = model.fit(
        train_ds,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=val_ds
    )
    t3 = time.time()
    print(f"✅ Phase 2 Duration: {(t3 - t2)/60:.2f} minutes")
    
    print(f"⏱️ Total Training Time: {(t3 - t0)/60:.2f} minutes")

    # 8. Save
    print(f"Saving model to {MODEL_SAVE_PATH}...")
    model.save(MODEL_SAVE_PATH)
    
    print(f"Saving labels to {LABELS_SAVE_PATH}...")
    with open(LABELS_SAVE_PATH, 'w') as f:
        for name in class_names:
            f.write(name + '\n')

    print("✅ Training complete.")

if __name__ == "__main__":
    main()