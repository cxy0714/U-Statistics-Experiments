#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

# ============================================================
# Detect GPU / CUDA Environment
# ============================================================
echo "Detecting runtime environment..."

HAS_GPU=false
CUDA_VER=""

if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    HAS_GPU=true
    # Extract CUDA version using python or nvidia-smi as a fallback
    CUDA_VER=$(python -c "import torch; print(torch.version.cuda)" 2>/dev/null || \
               nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+')
    echo "✅ GPU detected. CUDA Version: ${CUDA_VER}"
else
    echo "ℹ️  No GPU detected. Using CPU mode."
fi

# ============================================================
# Initialize Conda
# ============================================================
# Compatible with both miniconda and anaconda base paths
CONDA_BASE=$(conda info --base 2>/dev/null || echo "/root/miniconda3")
eval "$(${CONDA_BASE}/bin/conda shell.bash hook)"

ENV_NAME="U-stats"

# Check if the environment already exists to avoid redundant creation
if ! conda env list | grep -q "^${ENV_NAME} "; then
    echo "Creating Conda environment '${ENV_NAME}' (Python 3.12)..."
    conda create -y -n "${ENV_NAME}" python=3.12
else
    echo "Environment '${ENV_NAME}' already exists. Skipping creation."
fi

conda activate "${ENV_NAME}"

# ============================================================
# Install PyTorch
# ============================================================
# Check for existing valid torch+cuda installation to save time
EXISTING_CUDA=$(python -c "import torch; print(torch.version.cuda or '')" 2>/dev/null || echo "")

if [ "$HAS_GPU" = true ] && [ -n "$EXISTING_CUDA" ]; then
    echo "✅ Found PyTorch with CUDA ${EXISTING_CUDA}. Skipping torch installation."
elif [ "$HAS_GPU" = true ]; then
    # Parse version numbers for conditional installation
    CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d'.' -f1)
    CUDA_MINOR=$(echo "$CUDA_VER" | cut -d'.' -f2)

    if [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 8 ]; then
        echo "CUDA ${CUDA_VER} detected → Installing PyTorch nightly (cu128)"
        pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
    elif [ "$CUDA_MAJOR" -ge 12 ] && [ "$CUDA_MINOR" -ge 4 ]; then
        echo "CUDA ${CUDA_VER} detected → Installing PyTorch stable (cu124)"
        pip install torch --index-url https://download.pytorch.org/whl/cu124
    else
        echo "CUDA ${CUDA_VER} detected → Installing PyTorch stable (cu121)"
        pip install torch --index-url https://download.pytorch.org/whl/cu121
    fi
else
    echo "CPU Mode detected → Installing PyTorch (CPU-only)"
    pip install torch --index-url https://download.pytorch.org/whl/cpu
fi

# ============================================================
# Install Other Dependencies
# ============================================================
echo "Installing dependency packages..."
pip install --upgrade \
    u-stats \
    pandas \
    matplotlib \
    networkx \
    seaborn \
    scipy \
    igraph \
    psutil

# ============================================================
# Verification
# ============================================================
echo "Verifying installation..."
python -c "
import torch
print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU Device: {torch.cuda.get_device_name(0)}')
try:
    import u_stats
    print(f'u-stats Version: {u_stats.__version__}')
except Exception as e:
    print(f'Error importing u-stats: {e}')
"

echo "✅ Deployment complete!"