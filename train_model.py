import os
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm

def prepare_yolo_dataset(source_dir, output_dir, split_ratio=0.8):
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        print(f"❌ Error: Source directory '{source_dir}' not found.")
        return

    # Clean existing output directory
    if output_path.exists():
        print(f"🧹 Cleaning existing '{output_dir}'...")
        shutil.rmtree(output_path)
    
    # Create Train/Val root folders
    (output_path / 'train').mkdir(parents=True, exist_ok=True)
    (output_path / 'val').mkdir(parents=True, exist_ok=True)

    # Find classes
    classes = [d.name for d in source_path.iterdir() if d.is_dir()]
    print(f"🏷️  Found classes: {classes}")

    total_images = 0
    
    for class_name in classes:
        # Create class folders in train/val
        (output_path / 'train' / class_name).mkdir(exist_ok=True)
        (output_path / 'val' / class_name).mkdir(exist_ok=True)
        
        # Find all images recursively (handling date subfolders)
        class_dir = source_path / class_name
        # rglob('*') scans recursively
        images = [f for f in class_dir.rglob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        
        # Shuffle and split
        random.shuffle(images)
        split_idx = int(len(images) * split_ratio)
        train_imgs = images[:split_idx]
        val_imgs = images[split_idx:]
        
        # Link files
        print(f"   Processing '{class_name}': {len(train_imgs)} train, {len(val_imgs)} val")
        
        for img in tqdm(train_imgs, desc=f"Linking {class_name} (Train)", leave=False):
            dest = output_path / 'train' / class_name / img.name
            # Handle duplicate filenames in recursive folders by prepending parent folder name
            if dest.exists():
                dest = output_path / 'train' / class_name / f"{img.parent.name}_{img.name}"
            os.symlink(img, dest)
            
        for img in tqdm(val_imgs, desc=f"Linking {class_name} (Val)  ", leave=False):
            dest = output_path / 'val' / class_name / img.name
            if dest.exists():
                dest = output_path / 'val' / class_name / f"{img.parent.name}_{img.name}"
            os.symlink(img, dest)
            
        total_images += len(images)

    print(f"\n✅ Done! Prepared {total_images} images in '{output_dir}'.")
    print(f"🚀 You can now train YOLO with: yolo classify train data={output_dir} model=yolov8n-cls.pt epochs=20")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare recursive dataset for YOLO classification")
    # Defaulting to your specific path
    parser.add_argument('--source', type=str, default='/mnt/f/MLClouds_incoming/resized/', help='Source folder containing classes')
    parser.add_argument('--output', type=str, default='yolo_dataset', help='Output folder for YOLO structure')
    
    args = parser.parse_args()
    
    prepare_yolo_dataset(args.source, args.output)