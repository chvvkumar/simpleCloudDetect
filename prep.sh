#!/bin/bash

# --- Configuration ---
VENV_NAME="venv"
PYTHON="python3"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}--- Setting up TensorFlow Environment for RTX 5070 Ti (WSL) ---${NC}"

# 1. Check for Python
if ! command -v $PYTHON &> /dev/null; then
    echo -e "${RED}Error: $PYTHON could not be found.${NC}"
    exit 1
fi

# 2. Clean up old environment
if [ -d "$VENV_NAME" ]; then
    echo -e "${YELLOW}Removing existing virtual environment...${NC}"
    rm -rf "$VENV_NAME"
fi

# 3. Create new venv
echo -e "${GREEN}Creating new virtual environment...${NC}"
$PYTHON -m venv $VENV_NAME

# 4. Activate venv
source $VENV_NAME/bin/activate

# 5. Upgrade pip (Critical for new wheels)
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# 6. Install TensorFlow with CUDA bundles
# We use standard tensorflow[and-cuda] first. 
# If you are on bleeding edge hardware, this is usually safer than nightly unless standard fails hard.
echo -e "${GREEN}Installing TensorFlow[and-cuda] and dependencies...${NC}"
pip install "tensorflow[and-cuda]" matplotlib numpy

# 7. Generate Activation Script with LD_LIBRARY_PATH
# This is the MAGIC STEP that fixes "Cannot dlopen some GPU libraries"
echo -e "${GREEN}Configuring CUDA paths...${NC}"

SITE_PACKAGES=$($PYTHON -c "import site; print(site.getsitepackages()[0])")
cat > run_with_gpu.sh << EOF
#!/bin/bash
# Source the venv
source $(pwd)/$VENV_NAME/bin/activate

# Add all nvidia pip-installed libs to LD_LIBRARY_PATH
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cuda_cupti/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cuda_nvcc/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cufft/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/curand/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cusolver/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cusparse/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/nvjitlink/lib

# Reduce OneDNN verbosity
export TF_ENABLE_ONEDNN_OPTS=0

# Execute the command passed to this script
exec "\$@"
EOF

chmod +x run_with_gpu.sh

echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${YELLOW}IMPORTANT: Ensure your Windows NVIDIA Driver is fully updated via GeForce Experience.${NC}"
echo -e "To train, run this command:"
echo -e "${GREEN}./run_with_gpu.sh python train.py --data_dir /mnt/f/MLClouds_incoming/resized/${NC}"
```

### How to use it:

1.  **Run the setup:**
    ```bash
    chmod +x setup_gpu_env.sh
    ./setup_gpu_env.sh