import json
import os
from datetime import datetime
from u_stats import UStats

M_MIN = 2
M_MAX = 12
N = 10**4
OUTPUT_DIR = "experiments/hoif/results"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, f"count_complexity_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def hoif_expression(m):
    return [[i, i + 1] for i in range(m - 1)]


results = []

for m in range(M_MIN, M_MAX + 1):
    print(f"m={m} ...", flush=True)

    ustat = UStats(expression=hoif_expression(m))

    row = {"m": m}
    for dediag in [True, False]:
        scaling, flops, memory = ustat.complexity(n=N, _dediag=dediag)
        key = "dediag" if dediag else "no_dediag"
        row[key] = {"scaling": scaling, "flops": flops, "memory": memory}
        print(f"  {key}: scaling={scaling}, flops={flops:.6e}, memory={memory}")

    results.append(row)

with open(OUTPUT_FILE, "w") as f:
    json.dump(results, f, indent=6, default=float)

print(f"\nSaved to {OUTPUT_FILE}")