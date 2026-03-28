#!/bin/bash
#SBATCH --job-name=count_hoif
#SBATCH --partition=64c512g # 
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --output=output/count_hoif_%j.out
#SBATCH --error=output/count_hoif_%j.err 


source ~/.bashrc

module load miniconda3

conda activate U-stats

echo "--- Starting Complexity Benchmark: $(date) ---"
python experiments/hoif/run_count_complexity.py

echo "--- Starting U2V Counting: $(date) ---"
python experiments/hoif/run_count_u2v.py

echo "--- All Tasks Finished: $(date) ---"