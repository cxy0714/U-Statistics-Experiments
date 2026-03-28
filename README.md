# Reproducibility Code for  
**On Computing and the Complexity of Computing Higher-Order U-Statistics, Exactly**  
**Authors:** Xingyu Chen, Ruiqi Zhang, Lin Liu  
**Link:** https://arxiv.org/abs/2508.12627  

---

## Overview

This repository provides the code for reproducing the experimental results in our paper *“On Computing and the Complexity of Computing Higher-Order U-Statistics, Exactly”*.  

Our implementation is based on the Python package `u-stat`, available on PyPI:  
- **U-Statistics (Python):** https://github.com/zrq1706/U-Statistics-python  

We also provide a corresponding R interface based on Python package `u-stat` :  
- **U-Statistics (R interface):** https://github.com/cxy0714/U-Statistics-R  
---

## Environment Setup

To install all dependencies, please refer to:

```bash
sh/env_update.slurm
```

---

## Section 4.1: Higher-Order Influence Functions (HOIF)

Here we focus on computing the main component of the HOIF estimators.  
For a complete implementation, please refer to our R package:  
`https://github.com/cxy0714/HOIF`

### Table 1

- **Main script:**  
  `experiments/hoif/run.py`

- **Execution script:**  
  `sh/run_hoif.slurm`

- **Output results:**  
  `experiments/hoif/results/benchmark_20260327_155952.json`

- **Table generation:**  
  `experiments/hoif/table.py`

---

### Table 6

- **Main scripts:**  
  `experiments/hoif/run_count_complexity.py`  
  `experiments/hoif/run_count_u2v.py`

- **Execution script:**  
  `sh/run_hoif_count.slurm`

- **Output results:**  
  `count_complexity_20260328_150321.json`  
  `count_u2v_20260328_153058.json`

---

## Section 4.2: Motif Counts

(To be added)

---

## Section 4.3: Distance Covariance (dCov)

### Table 3

#### Our U-Statistics Implementation

- **Main script:**  
  `experiments/dcov/run.py`

- **Execution script:**  
  `sh/run_dcov_ustat.slurm`

- **Output results:**  
  `dcov_results_numpy_20250817_043111.json`  
  `dcov_results_torch_20250817_041908.json`

---

#### Shao's MATLAB Implementation

- **Complete U-statistics:**  
  `data/stock_market/only_dcov.m`

- **Randomized U-statistics:**  
  `data/stock_market/randmized.m`

- **Output results:**  
  `data/stock_market/result/`

---

## Notes

- All experiments are designed to run on a SLURM-based cluster environment.  
- Please adjust paths and resource configurations according to your system if needed.  
