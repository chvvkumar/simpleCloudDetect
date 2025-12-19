#!/bin/bash

# 1. Update System & Install Python Dependencies
echo "Updating system..."
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv

# 2. Create Project Directory
PROJECT_DIR=~/simpleCloudDetect_Modern
mkdir -p "$PROJECT_DIR"
cd "$PROJECT_DIR"

# 3. Create Virtual Environment
echo "Creating virtual environment..."
python3 -m venv venv

# 4. Activate Environment
source venv/bin/activate

# 5. Install PyTorch with CUDA Support (for NVIDIA GPU)
# Note: Assumes CUDA drivers are installed on Windows host
echo "Installing PyTorch (CUDA)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 6. Install Other Utilities
echo "Installing utilities..."
pip install onnx onnxruntime pillow numpy requests

# 7. Create Symlink to Dataset (Optional but convenient)
# This lets you refer to 'dataset' instead of the full path
if [ -d "/mnt/f/MLClouds_incoming/resized/" ]; then
    echo "Linking dataset..."
    ln -s /mnt/f/MLClouds_incoming/resized/ dataset
else
    echo "Warning: Dataset path /mnt/f/MLClouds_incoming/resized/ not found."
fi

echo "Setup complete! Activate via: source ~/simpleCloudDetect_Modern/venv/bin/activate"
