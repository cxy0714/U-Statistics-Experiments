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

To install all dependencies on a SLURM-based CPU cluster, please refer to [`sh/env_update.slurm`](sh/env_update.slurm). To install on a single GPU machine, please refer to [`sh/gpu_setup.sh`](sh/gpu_setup.sh).

---

## Section 4.1: Higher-Order Influence Functions (HOIF)

All scripts and results for this section are in [`experiments/hoif/`](experiments/hoif/).

This section focuses on computing the main component of the HOIF estimators.  
For a complete implementation, please refer to our R package:  
https://github.com/cxy0714/HOIF

### Table 1

- **Main script:** `run.py`
- **Execution script:** `sh/run_hoif.slurm` (on CPU cluster); manual execution on single GPU machine.
- **Output results:** `results/benchmark_20260327_155952.json`, `results/benchmark_20260328_195135.json`
- **Table generation:** `table.py` → `results/benchmark_*_summary.txt`

---

### Table 6

- **Main scripts:** `run_count_complexity.py`, `run_count_u2v.py`
- **Execution script:** `sh/run_hoif_count.slurm` (on CPU cluster); manual execution on single GPU machine.
- **Output results:** `results/count_complexity_20260328_150321.json`, `results/count_u2v_20260328_153058.json`, `results/count_hoif_57008502.out`

---

## Section 4.2: Motif Counts

All scripts and results for this section are in [`experiments/motif_count/`](experiments/motif_count/).

### Tables 2 and 4

Dependencies: [Peregrine](https://github.com/pdclab/peregrine) and [igraph](https://igraph.org/). Peregrine installation instructions are available at the link above; igraph is installed via pip in [`sh/env_update.slurm`](sh/env_update.slurm).

- **Main script:** `run.py`
- **Execution scripts:** `sh/run_motif_fast.slurm` (Peregrine and our method only), `sh/run_motif_igraph.slurm` (igraph only)
- **Output results:** `results/benchmark_size{3,4}_{fast,igraph}_*.json`
- **Table generation:** `run_table_cpu.py` → `results/table_size{3,4}.txt`

### Table 5

Dependencies: [cuGraph](https://docs.rapids.ai/api/cugraph/stable/). Installation instructions are available at the link above.

- **Main script:** `run_cugraph.py`
- **Execution:** Manual execution on single GPU machine.
- **Output results:** `results/GPU_triangle_benchmark_20251115_162807.json` → `triangle_comparison_table.txt`

---

## Section 4.3: Distance Covariance (dCov)

All scripts and results for this section are in [`experiments/dcov/`](experiments/dcov/).

### Table 3

#### Our U-Statistics Implementation

- **Main script:** `run.py` (execution) and `kernel.py` (data processing)
- **Execution script:** `sh/run_dcov_ustat.slurm`
- **Output results:** `results/dcov_results_{numpy,torch}_20250817_*.json`

#### Shao's MATLAB Implementation

All scripts and results are in [`data/stock_market/`](data/stock_market/).

- **Complete U-statistics:** `only_dcov.m`
- **Randomized U-statistics:** `randmized.m`
- **Output results:** `result/`

---

## Notes

- CPU-based experiments were run on the π 2.0 and Siyuan-1 clusters supported by the Center for High Performance Computing at Shanghai Jiao Tong University. GPU-based experiments were run on a single GPU machine.
- Please adjust paths and resource configurations as needed for your system.