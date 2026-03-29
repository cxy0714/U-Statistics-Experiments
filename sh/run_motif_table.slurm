#!/bin/bash

#SBATCH --job-name=motif_table
#SBATCH --partition=cpu
#SBATCH -N 1         
#SBATCH --ntasks-per-node=1
#SBATCH --output=output/motif_table_%j.out
#SBATCH --error=output/motif_table_%j.err 


source ~/.bashrc


module load miniconda3/24.3.0


source /lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-8.5.0/miniconda3-24.3.0-3zpgys4jvd5jttc4curqobgdnwpkxawb/etc/profile.d/conda.sh

conda activate u_251117


python experiments/motif_count/run_table_cpu.py

python experiments/motif_count/run_table_gpu.py