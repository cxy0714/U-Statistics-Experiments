#!/bin/bash
#SBATCH -J matlab_dcov
#SBATCH -p 64c512g
#SBATCH --output=output/matlab_%A_%a.out
#SBATCH --error=output/matlab_%A_%a.err
#SBATCH -n 1
#SBATCH --ntasks-per-node=64

module load matlab/r2025a
cd ~/alg-u-experiments-dcov/data/stock_market


al_value=1.5
matlab -nodesktop -nosplash -r "al=${al_value}; randmized; exit;"

