#!/bin/bash
#SBATCH --job-name=table
#SBATCH --partition=64c512g # 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output/table_%j.out
#SBATCH --error=output/table_%j.err 


source ~/.bashrc

module load miniconda3

conda activate U-stats

python experiments/hoif/table.py