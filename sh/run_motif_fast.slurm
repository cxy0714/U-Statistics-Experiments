#!/bin/bash

#SBATCH --job-name=motif_fast
#SBATCH --partition=cpu
#SBATCH -N 1         
#SBATCH --ntasks-per-node=40
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err 

# 1. 首先设置关键的库路径（避免被bashrc覆盖）
export TBBROOT=/lustre/home/acct-insll/insll-xingyuchen/alg-u-experiments-main/Peregrine/tbb2020
export LD_LIBRARY_PATH=$TBBROOT/lib/intel64/gcc4.8:$LD_LIBRARY_PATH

# 2. 然后source bashrc（可能包含其他重要设置）
source ~/.bashrc

# 3. 加载模块
module load miniconda3/24.3.0

# 4. 初始化conda
source /lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-8.5.0/miniconda3-24.3.0-3zpgys4jvd5jttc4curqobgdnwpkxawb/etc/profile.d/conda.sh

# 5. 激活环境
conda activate u_251117

# 6. 运行程序
python experiments/motif_count/run.py --mode "fast" --verify