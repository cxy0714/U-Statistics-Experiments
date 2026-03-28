"""
Count partitions satisfying the U-statistics condition from Algorithm 1.

For a given calA (collection of "take sets"), count partitions pi in Part(m) such that:
    forall Q in pi, forall A in calA: |Q ∩ A| < 2
i.e., every block of pi intersects every set in calA in at most 1 element.

To add a new experiment, define a CalAFactory (a callable m -> list[frozenset])
and register it in EXPERIMENTS below.
"""

import json
import os
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from typing import Callable

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

OUTPUT_FILE = f"experiments/hoif/results/count_u2v_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# ---------------------------------------------------------------------------
# CalA factories
# A CalAFactory is a callable:  m -> list[frozenset[int]]
# Elements are drawn from {1, ..., m}.
# ---------------------------------------------------------------------------

CalAFactory = Callable[[int], list[frozenset]]


def calA_consecutive_pairs(m: int) -> list[frozenset]:
    """
    A = { {1,2}, {2,3}, ..., {m-1,m} }
    A chain of consecutive pairs covering all m indices.
    Experimentally: valid partition count == Bell(m-1).
    """
    return [frozenset({i, i + 1}) for i in range(1, m)]


# ---------------------------------------------------------------------------
# Register experiments here.
# Each entry: (name, description, factory, m_range)
# To add a new calA type, append a new tuple.
# ---------------------------------------------------------------------------

EXPERIMENTS: list[tuple[str, str, CalAFactory, range]] = [
    (
        "consecutive_pairs",
        "calA = {{i, i+1} : i=1..m-1}  (chain of adjacent pairs)",
        calA_consecutive_pairs,
        range(2, 13),
    ),
    # ---- Add new experiment configurations below ----
    # (
    #     "all_pairs",
    #     "calA = all pairs from {1,...,m}",
    #     lambda m: [frozenset(p) for p in combinations(range(1, m+1), 2)],
    #     range(2, 13),
    # ),
]

# ---------------------------------------------------------------------------
# Core combinatorics
# ---------------------------------------------------------------------------


def bell_number(n: int) -> int:
    """Compute the n-th Bell number via the Bell triangle."""
    if n == 0:
        return 1
    triangle = [1]
    for _ in range(n - 1):
        new_row = [triangle[-1]]
        for j in range(len(triangle)):
            new_row.append(new_row[-1] + triangle[j])
        triangle = new_row
    return triangle[-1]


def block_satisfies_condition(block: frozenset, calA: list[frozenset]) -> bool:
    """True iff the block intersects every set in calA in at most 1 element."""
    return all(len(block & A) < 2 for A in calA)


def generate_partitions(elements: list):
    """
    Generate all set partitions of `elements` as lists of frozensets,
    via restricted growth strings (lexicographic order).
    """
    n = len(elements)
    if n == 0:
        yield []
        return

    assignment = [0] * n
    max_so_far = [0] * n

    def backtrack(pos: int):
        if pos == n:
            d = defaultdict(list)
            for i, b in enumerate(assignment):
                d[b].append(elements[i])
            yield [frozenset(v) for v in d.values()]
            return
        upper = (max_so_far[pos - 1] + 1) if pos > 0 else 0
        for b in range(upper + 1):
            assignment[pos] = b
            max_so_far[pos] = max(max_so_far[pos - 1] if pos > 0 else 0, b)
            yield from backtrack(pos + 1)

    yield from backtrack(0)


def count_valid_partitions(m: int, calA: list[frozenset]) -> int:
    """
    Count partitions pi of {1,...,m} satisfying:
        forall Q in pi, forall A in calA: |Q ∩ A| < 2
    """
    elements = list(range(1, m + 1))
    return sum(
        1
        for partition in generate_partitions(elements)
        if all(block_satisfies_condition(block, calA) for block in partition)
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_all_experiments() -> None:
    all_results = []

    for exp_name, exp_desc, factory, m_range in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f"Experiment : {exp_name}")
        print(f"Description: {exp_desc}")
        print(f"{'='*60}")
        print(f"{'m':>4} | {'valid partitions':>18} | {'Bell(m)':>12} | {'ratio':>8}")
        print("-" * 52)

        rows = []
        for m in m_range:
            calA = factory(m)
            valid = count_valid_partitions(m, calA)
            bell = bell_number(m)
            ratio = valid / bell

            rows.append({
                "m": m,
                "calA": [sorted(A) for A in calA],
                "valid_partitions": valid,
                "bell_number": bell,
                "ratio": round(ratio, 6),
            })
            print(f"{m:>4} | {valid:>18} | {bell:>12} | {ratio:>8.4f}")

        all_results.append({
            "experiment": exp_name,
            "description": exp_desc,
            "rows": rows,
        })

    # Write output
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    payload = {
        "description": (
            "Valid partition counts for the U-stat condition: "
            "forall block Q in pi, forall A in calA: |Q ∩ A| < 2."
        ),
        "generated_at": datetime.now().isoformat(),
        "experiments": all_results,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_all_experiments()