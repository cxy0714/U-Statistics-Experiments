#!/bin/bash
#SBATCH --job-name=hoif
#SBATCH --partition=64c512g # 
#SBATCH -N 1
#SBATCH --ntasks-per-node=64
#SBATCH --output=output/hoif_%j.out
#SBATCH --error=output/hoif_%j.err 


source ~/.bashrc

module load miniconda3

conda activate U-stats

python experiments/hoif/run.py