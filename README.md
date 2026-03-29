# Reproducibility Code for  
**On Computing and the Complexity of Computing Higher-Order U-Statistics, Exactly**  
**Authors:** Xingyu Chen, Ruiqi Zhang, Lin Liu  
**Link:** https://arxiv.org/abs/2508.12627  

---

## Overview

This repository provides code for reproducing the experimental results in our paper *"On Computing and the Complexity of Computing Higher-Order U-Statistics, Exactly"*.

Our implementation is based on the Python package `u-stat`, available on PyPI:  
- **U-Statistics (Python):** https://github.com/zrq1706/U-Statistics-python  

We also provide a corresponding R interface:  
- **U-Statistics (R interface):** https://github.com/cxy0714/U-Statistics-R  

---

## Environment Setup

To install all dependencies on a SLURM-based CPU cluster, please refer to `sh/env_update.slurm`. To install on a single GPU machine, please refer to `sh/gpu_setup.sh`.

---

## Section 4.1: Higher-Order Influence Functions (HOIF)

This section focuses on computing the main component of the HOIF estimators.  
For a complete implementation, please refer to our R package:  
https://github.com/cxy0714/HOIF

### Table 1

- **Main script:** `experiments/hoif/run.py`

- **Execution script:** `sh/run_hoif.slurm` (on CPU cluster); manual execution on single GPU machine.

- **Output results:**  
  `experiments/hoif/results/benchmark_20260327_155952.json`  
  `experiments/hoif/results/benchmark_20260328_195135.json`

- **Table generation scripts and summaries:**  
  `experiments/hoif/table.py`  
  `experiments/hoif/results/benchmark_20260327_155952_summary.txt`  
  `experiments/hoif/results/benchmark_20260328_195135_summary.txt`

---

### Table 6

- **Main scripts:**  
  `experiments/hoif/run_count_complexity.py`  
  `experiments/hoif/run_count_u2v.py`

- **Execution script:** `sh/run_hoif_count.slurm` (on CPU cluster); manual execution on single GPU machine.

- **Output results:**  
  `experiments/hoif/results/count_complexity_20260328_150321.json`  
  `experiments/hoif/results/count_u2v_20260328_153058.json`  
  `experiments/hoif/results/count_hoif_57008502.out`

---

## Section 4.2: Motif Counts

### Tables 2 and 4

Dependencies: [Peregrine](https://github.com/pdclab/peregrine) and [igraph](https://igraph.org/). Peregrine installation instructions are available at the link above; igraph is installed via pip in `sh/env_update.slurm`.

- **Main script:** `experiments/motif_count/run.py`

- **Execution scripts:**  
  `sh/run_motif_fast.slurm` (Peregrine and our method only)  
  `sh/run_motif_igraph.slurm` (igraph only)

- **Output results:**  
  `experiments/motif_count/results/benchmark_size3_fast_20260328_143440.json`  
  `experiments/motif_count/results/benchmark_size3_igraph_20260328_143404.json`  
  `experiments/motif_count/results/benchmark_size4_fast_20260328_173329.json`  
  `experiments/motif_count/results/benchmark_size4_igraph_20260328_155723.json`

- **Table generation scripts and outputs:**  
  `experiments/motif_count/run_table_cpu.py`  
  `experiments/motif_count/results/table_size3.txt`  
  `experiments/motif_count/results/table_size4.txt`

### Table 5

Dependencies: [cuGraph](https://docs.rapids.ai/api/cugraph/stable/). Installation instructions are available at the link above.

- **Main script:** `experiments/motif_count/run_cugraph.py`

- **Execution:** Manual execution on single GPU machine.

- **Output results:**  
  `experiments/motif_count/results/GPU_triangle_benchmark_20251115_162807.json`

- **Table output:**  
  `experiments/motif_count/triangle_comparison_table.txt`

---

## Section 4.3: Distance Covariance (dCov)

### Table 3

#### Our U-Statistics Implementation

- **Main script:** `experiments/dcov/run.py`

- **Execution script:** `sh/run_dcov_ustat.slurm`

- **Output results:**  
  `dcov_results_numpy_20250817_043111.json`  
  `dcov_results_torch_20250817_041908.json`

---

#### Shao's MATLAB Implementation

- **Complete U-statistics:** `data/stock_market/only_dcov.m`

- **Randomized U-statistics:** `data/stock_market/randmized.m`

- **Output results:** `data/stock_market/result/`

---

## Notes

- CPU-based experiments were run on the π 2.0 and Siyuan-1 clusters supported by the Center for High Performance Computing at Shanghai Jiao Tong University. GPU-based experiments were run on a single GPU machine.
- Please adjust paths and resource configurations as needed for your system.