#!/bin/bash
#SBATCH -J matlab_dcov
#SBATCH -p 64c512g
#SBATCH --output=output/matlab_%A_%a.out
#SBATCH --error=output/matlab_%A_%a.err
#SBATCH -n 1
#SBATCH --cpus-per-task 64

module load matlab/r2025a
cd ~/alg-u-experiments-dcov/data/stock_market



matlab -nodesktop -nosplash -r "only_dcov; exit;"

