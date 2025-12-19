#!/bin/bash

SOURCE_DIR="/mnt/f/MLClouds_incoming/resized"
DEST_DIR="$HOME/fast_dataset"

echo "----------------------------------------------------------------"
echo "🚀 Optimization: Moving Data to Native Linux Filesystem"
echo "----------------------------------------------------------------"
echo "Why: Reading files from /mnt/f/ (Windows) is slow in WSL."
echo "     Reading files from $HOME (Linux) is instant."
echo "----------------------------------------------------------------"

if [ -d "$DEST_DIR" ]; then
    echo "✅ Fast dataset folder already exists at $DEST_DIR"
else
    echo "📦 Copying images from Windows drive... (This happens once)"
    echo "   Source: $SOURCE_DIR"
    echo "   Dest:   $DEST_DIR"
    
    # Create directory
    mkdir -p "$DEST_DIR"
    
    # Copy with progress bar (if rsync installed) or cp
    if command -v rsync &> /dev/null; then
        rsync -ah --info=progress2 "$SOURCE_DIR/" "$DEST_DIR/"
    else
        cp -r "$SOURCE_DIR/"* "$DEST_DIR/"
    fi
    echo "✅ Copy complete."
fi

# Update the symlink in the project folder
if [ -L "dataset" ]; then
    echo "🔗 Updating 'dataset' symlink to point to fast storage..."
    rm dataset
    ln -s "$DEST_DIR" dataset
else
    echo "🔗 Creating 'dataset' symlink..."
    ln -s "$DEST_DIR" dataset
fi

echo "----------------------------------------------------------------"
echo "🎉 Done! Run 'python train_model.py' again."
echo "   Your data loading times should drop significantly."
echo "----------------------------------------------------------------"