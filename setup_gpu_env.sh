#!/bin/bash

# --- Configuration ---
VENV_NAME="venv"
PYTHON="python3"

# --- Colors ---
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}--- Setting up TensorFlow Nightly for RTX 5070 Ti (WSL) ---${NC}"

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

# 5. Upgrade pip
echo -e "${GREEN}Upgrading pip...${NC}"
pip install --upgrade pip

# 6. Install tf-nightly (REQUIRED for RTX 50-series)
echo -e "${GREEN}Installing tf-nightly[and-cuda]...${NC}"
pip uninstall -y tensorflow
pip install "tf-nightly[and-cuda]" matplotlib numpy

# 7. Generate Activation Script with LD_LIBRARY_PATH
echo -e "${GREEN}Configuring CUDA paths...${NC}"

cat > run_with_gpu.sh << EOF
#!/bin/bash
# Source the venv
source $(pwd)/$VENV_NAME/bin/activate

# --- CRITICAL: Add ALL NVIDIA Library Paths ---
# Added: cuda_nvrtc (Required for runtime compilation)
# Added: /usr/lib/wsl/lib (Required for WSL2 libcuda.so)

export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/usr/lib/wsl/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cudnn/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cublas/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cuda_cupti/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cuda_nvcc/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cuda_nvrtc/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cuda_runtime/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cufft/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/curand/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cusolver/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/cusparse/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/nccl/lib
export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:$(pwd)/$VENV_NAME/lib/python*/site-packages/nvidia/nvjitlink/lib

# Reduce verbosity
export TF_CPP_MIN_LOG_LEVEL=2
export TF_ENABLE_ONEDNN_OPTS=0

# Execute the command passed to this script
exec "\$@"
EOF

chmod +x run_with_gpu.sh

echo -e "${GREEN}Setup Complete!${NC}"
echo -e "${YELLOW}IMPORTANT: We fixed the missing NVRTC path. Run the command below to test.${NC}"
echo -e "${GREEN}./run_with_gpu.sh python -c \"import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))\"${NC}"