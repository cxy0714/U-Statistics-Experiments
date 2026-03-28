#!/bin/bash

#SBATCH --partition=64c512g
#SBATCH -n 1
#SBATCH --ntasks-per-node=64
#SBATCH --output=output/slurm_%A_%a.out
#SBATCH --error=output/slurm_%A_%a.err
#SBATCH --exclusive   

# 获取传入的参数

module load miniconda3
eval "$(conda shell.bash hook)"  
conda activate U-stats
echo "Activated Conda environment: $(which python)"

# 运行模拟
python experiments/dcov/run.py
