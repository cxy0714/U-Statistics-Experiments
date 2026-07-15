#!/bin/bash

#SBATCH --job-name=hoif_pathcx
#SBATCH --partition=cpu        # cpu queue
#SBATCH -n 40                  # 40 cores total (a full single node)
#SBATCH --ntasks-per-node=40   # cores per node
#SBATCH --exclusive            # own the whole node for its large memory (~190GB)
#SBATCH --time=7-00:00:00      # partition cap; no real time limit: memory is the only gate
#SBATCH --output=output/%j.out
#SBATCH --error=output/%j.err

# 2. Source bashrc (may contain other important settings)
source ~/.bashrc

# 3. Load modules
module load miniconda3/24.3.0

# 4. Initialize conda
source /lustre/opt/cascadelake/linux-rhel8-skylake_avx512/gcc-8.5.0/miniconda3-24.3.0-3zpgys4jvd5jttc4curqobgdnwpkxawb/etc/profile.d/conda.sh

# 5. Activate environment
conda activate u_251117

# 6. Run the program
#   m=7, n=1000/4000/10000, 3 timing repeats each, float32, all 40 cores (--threads 40).
#   No time limit: long runs necessarily have high order and large intermediate
#   tensors, so they hit the memory gate (--mem-limit-gb 180) first and are marked
#   skipped(oom). timeout is set very large, i.e. memory is the only backstop.
python experiments/hoif/run_path_complexity_full.py \
    --m 7\
    --n 1000 2000 10000\
    --repeats 3 \
    --threads 40 \
    --mem-limit-gb 180 \
    --path-timeout 9999 \
    --exec-timeout 9000
