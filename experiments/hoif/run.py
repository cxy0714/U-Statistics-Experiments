import time
import json
import numpy as np
from datetime import datetime
from u_stats import ustat, set_backend

set_backend("torch")

SEED = 42
SIZES = [1000, 2000, 4000, 8000, 10000]
ORDERS = [2, 3, 4, 5, 6, 7]
NUM_RUNS = 10
OUTPUT_FILE = f"experiments/hoif/results/benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

print("Global warmup...")
_tensors = [np.random.rand(100, 100).astype(np.float64) for _ in range(2)]
ustat(tensors=_tensors, expression=[[1,2],[2,3]], average=True)

results = []

for order in ORDERS:
    mode = [[str(i), str(i + 1)] for i in range(1, order)]
    num_tensors = order - 1

    for size in SIZES:
        print(f"order={order}, size={size} ...", flush=True)

        # Generate data once per (order, size)
        np.random.seed(SEED)
        tensors = [np.random.rand(size, size).astype(np.float64) for _ in range(num_tensors)]

        # Warmup
        ustat(tensors=tensors, expression=mode, average=True)

        # 10 timed runs
        run_times = []
        run_values = []
        for _ in range(NUM_RUNS):
            t0 = time.perf_counter()
            val = ustat(tensors=tensors, expression=mode, average=True)
            run_times.append(time.perf_counter() - t0)
            run_values.append(float(val) if not isinstance(val, float) else val)

        results.append({
            "order": order,
            "size": size,
            "times": run_times,
            "values": run_values,
        })

        print(f"  avg={np.mean(run_times):.4f}s  val={run_values[0]:.6e}")

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nSaved to {OUTPUT_FILE}")