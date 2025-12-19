#!/bin/bash

# 1. Update System & Install Python Dependencies
echo "Updating system..."
sudo apt-get update && sudo apt-get install -y python3-pip python3-venv

# 2. Use current directory as project directory
PROJECT_DIR=$(pwd)
echo "Setting up environment in: $PROJECT_DIR"

# 3. Create Virtual Environment
echo "Creating virtual environment..."
python3 -m venv venv_pytorch

# 4. Activate Environment
source venv_pytorch/bin/activate

# 5. Install PyTorch with CUDA Support (for NVIDIA GPU)
# Note: Assumes CUDA drivers are installed on Windows host
echo "Installing PyTorch (CUDA 13.0)..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 6. Install Other Utilities
echo "Installing utilities..."
pip install onnx onnxruntime pillow numpy requests

# 7. Create Symlink to Dataset (Optional but convenient)
# This lets you refer to 'dataset' instead of the full path
if [ -d "/mnt/f/MLClouds_incoming/resized/" ]; then
    echo "Linking dataset..."
    ln -sf /mnt/f/MLClouds_incoming/resized/ dataset
else
    echo "Warning: Dataset path /mnt/f/MLClouds_incoming/resized/ not found."
fi

echo "Setup complete!"
echo "To use this environment, run: source venv_pytorch/bin/activate"
echo "Then you can train with: python train_model.py"