#!/bin/bash

#SBATCH --job-name=motif_fast
#SBATCH --partition=cpu
#SBATCH -N 1         
#SBATCH --ntasks-per-node=40
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err 


export TBBROOT=Peregrine/tbb2020
export LD_LIBRARY_PATH=$TBBROOT/lib/intel64/gcc4.8:$LD_LIBRARY_PATH


source ~/.bashrc


module load miniconda3/24.3.0


source /lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-8.5.0/miniconda3-24.3.0-3zpgys4jvd5jttc4curqobgdnwpkxawb/etc/profile.d/conda.sh


conda activate u_251117


python experiments/motif_count/run.py --mode "fast" --verify