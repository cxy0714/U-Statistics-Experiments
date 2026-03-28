#!/bin/bash
#SBATCH --job-name=create_and_install_env
#SBATCH --partition=64c512g 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=create_env_%j.log

# 1. Load the miniconda3 module
module load miniconda3

# 2. Initialize Conda to ensure subsequent commands execute correctly
# This is crucial for making the 'conda' command available in non-interactive shells
eval "$(conda shell.bash hook)"

# 3. Define your Conda environment name
ENV_NAME="U-stats"

# 4. Check if the environment exists. If not, create it.
# The -y flag automatically confirms installation
echo "Checking Conda environment: '${ENV_NAME}'..."
if ! conda env list | grep -q "${ENV_NAME}"; then
    echo "Environment '${ENV_NAME}' not found. Creating it now..."
    conda create -y -n "${ENV_NAME}" python=3.13 # Specifying a Python version is recommended
    echo "Environment '${ENV_NAME}' created successfully."
else
    echo "Environment '${ENV_NAME}' already exists. Skipping creation."
fi

# 5. Activate the Conda environment
conda activate "${ENV_NAME}"
echo "Currently active environment: $(conda env list | grep '*' | awk '{print $1}')"

# 6. Install or update required packages
# You can use a requirements.txt file or list packages directly
echo "Starting package installation/update..."

# It is generally recommended to use pip for these packages unless 
# a specific conda version is required for compatibility
python -m pip install --upgrade u-stats
pip install torch pandas matplotlib networkx seaborn scipy igraph psutil

echo "All packages installed and updated successfully!"

# Optional: List installed packages for debugging and verification
pip list

echo "Slurm job completed."