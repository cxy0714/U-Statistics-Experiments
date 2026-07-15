"""
Full per-(m, n) path-strategy report for the HOIF U-statistic.

For each U-statistic order m and sample size n, and for each opt_einsum path
optimizer, this records in one place both the STATIC estimates and the ACTUAL
execution cost:

    * path_find_time  : wall-clock time to FIND the contraction paths, summed
                        over the Bell(m-1) subexpressions (mean over repeats)
    * exec_time       : wall-clock time to actually COMPUTE the U-statistic via
                        UStats.compute(..., optimize=opt) (mean over repeats)
    * flops           : estimated FLOPs of the chosen paths (summed, as in
                        UStats.complexity)
    * peak_mem_elems  : estimated largest intermediate tensor (max, # elements)
    * scaling         : overall leading exponent s in O(n^s) (max)
    * n^2 / n^3 / ... : the complexity-order distribution -- how many of the
                        Bell(m-1) subexpressions fall into each order O(n^s)

Both the path-finding pass and the execution pass are repeated `--repeats`
times and averaged. Each optimizer's two passes run in an isolated subprocess so
that (a) a runaway path search or a slow contraction can be killed on a
wall-clock timeout, and (b) an out-of-memory contraction cannot crash the whole
run. Execution is additionally guarded by a real-memory ceiling: the contraction is
launched and its actual resident memory (RSS) is polled while it runs; if usage
crosses `--mem-limit-gb` the process is killed and the cell is marked
`skipped(oom)`. The estimate is never used to pre-skip -- it is only reported.

Backend: torch on CPU (CUDA is disabled up-front). Default dtype is float32 to
match the paper's run.py and because it is ~2.6x faster than float64 at large n;
pass --dtype float64 for higher-precision accumulation.

One .txt table is written per (m, n) -- one row per optimizer -- plus a single
JSON with all raw numbers.

Examples
--------
    python experiments/hoif/run_path_complexity_full.py --m 4 6 8 --n 1000 5000
    python experiments/hoif/run_path_complexity_full.py \
        --m 4 5 6 --n 2000 --repeats 10 --optimizers greedy dp optimal \
        --mem-limit-gb 8 --exec-timeout 300
"""

import os

# Force CPU torch BEFORE any torch import (also propagates to spawned workers,
# which re-execute this module top-level under the "spawn" start method).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Pin BLAS/OpenMP thread counts up-front so torch/MKL do NOT spawn one thread
# per physical core. On a many-core cluster node where SLURM only grants a
# subset of cores, the default (threads = all physical cores) massively
# oversubscribes the allocation and thrashes -- observed ~150s vs ~0.75s for a
# single m=4/n=4000 contraction. --threads overrides this at runtime; we set a
# conservative default here that is refined once args are parsed.
def _pin_threads(nthreads):
    """Set every relevant thread-count env var to `nthreads` (as a string)."""
    n = str(int(nthreads))
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[var] = n


# Default before args are parsed: honour SLURM's allocation if present,
# otherwise fall back to the machine's core count.
_default_threads = (
    os.environ.get("SLURM_CPUS_PER_TASK")
    or os.environ.get("SLURM_CPUS_ON_NODE")
    or os.cpu_count()
    or 1
)
_pin_threads(_default_threads)

import argparse
import json
import multiprocessing as mp
import time
from collections import Counter
from datetime import datetime

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

ALL_OPTIMIZERS = [
    "greedy",
    "dp",
    "branch-1",
    "branch-2",
    "branch-all",
    "optimal",
    "auto",
    "auto-hq",
    "random-greedy",
]
OUTPUT_DIR = "experiments/hoif/results"


def hoif_expression(m):
    """HOIF kernel of order m: a chain of (m-1) consecutive-index matrices."""
    return [[i, i + 1] for i in range(m - 1)]


# ---------------------------------------------------------------------------
# Subprocess workers (must be module-level so "spawn" can pickle them)
# ---------------------------------------------------------------------------


def _path_worker(q, m, n, optimizer, repeats, dediag):
    """Time path finding and aggregate the static complexity metrics."""
    try:
        import opt_einsum as oe
        from u_stats import UStats

        ustat = UStats(expression=hoif_expression(m))
        shapes = [(n,) * len(inputs) for inputs in ustat._inputs]
        subexpressions = list(ustat.get_all_subexpressions(dediag=dediag))

        times = []
        scaling = 0
        flops = 0.0
        largest = 0.0
        hist = Counter()
        for _ in range(repeats):
            t0 = time.perf_counter()
            sc, fl, lg, h = 0, 0.0, 0.0, Counter()
            for _, subexpression in subexpressions:
                _, info = oe.contract_path(
                    subexpression, *shapes, optimize=optimizer, shapes=True
                )
                s = int(max(info.scale_list))
                sc = max(sc, s)
                fl += float(info.opt_cost)
                lg = max(lg, float(info.largest_intermediate))
                h[s] += 1
            times.append(time.perf_counter() - t0)
            scaling, flops, largest, hist = sc, fl, lg, h

        q.put(
            {
                "ok": True,
                "times": times,
                "num_subexpressions": len(subexpressions),
                "num_operands": len(ustat._inputs),
                "scaling": scaling,
                "flops": flops,
                "largest": largest,
                "hist": {int(k): int(v) for k, v in hist.items()},
            }
        )
    except Exception as e:  # noqa: BLE001 -- report, don't crash the parent
        q.put({"ok": False, "error": repr(e)})


class _PeakRSSSampler:
    """Background thread that polls this process's RSS and keeps the maximum.

    torch CPU tensors are allocated in C++, so Python-level tracemalloc misses
    them; process RSS is the portable way to see the real footprint. The exec
    pass runs in its own subprocess, so RSS here reflects just this optimizer's
    contraction (tensors + intermediates), not the rest of the run.
    """

    def __init__(self, interval=0.005):
        import threading

        import psutil

        self._proc = psutil.Process()
        self._interval = interval
        self._peak = self._proc.memory_info().rss
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            try:
                rss = self._proc.memory_info().rss
                if rss > self._peak:
                    self._peak = rss
            except Exception:  # noqa: BLE001 -- process may be tearing down
                break
            self._stop.wait(self._interval)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()

    @property
    def peak(self):
        return self._peak


def _exec_worker(q, m, n, optimizer, repeats, dediag, seed, dtype_str, nthreads):
    """Actually compute the U-statistic; time it and measure peak RSS."""
    try:
        import numpy as np
        import psutil
        import torch
        from u_stats import UStats, set_backend

        # Explicit thread pinning: env vars alone can be ignored once MKL is
        # loaded. Without this, torch defaults to one thread per physical core,
        # oversubscribing a partial SLURM allocation and running ~200x slower.
        torch.set_num_threads(int(nthreads))
        set_backend("torch")  # CPU: CUDA is hidden via CUDA_VISIBLE_DEVICES
        dtype = np.float64 if dtype_str == "float64" else np.float32

        np.random.seed(seed)
        num_tensors = m - 1
        tensors = [np.random.rand(n, n).astype(dtype) for _ in range(num_tensors)]

        ustat = UStats(expression=hoif_expression(m))

        # Warmup (torch lazy init, path caching) -- not timed.
        val = ustat.compute(tensors, average=True, _dediag=dediag, optimize=optimizer)

        # RSS with input tensors resident but before the contraction: subtract
        # this to report the contraction's own peak (intermediates + output).
        baseline_rss = psutil.Process().memory_info().rss

        times = []
        peak_rss = baseline_rss
        for _ in range(repeats):
            with _PeakRSSSampler() as sampler:
                t0 = time.perf_counter()
                val = ustat.compute(
                    tensors, average=True, _dediag=dediag, optimize=optimizer
                )
                times.append(time.perf_counter() - t0)
            peak_rss = max(peak_rss, sampler.peak)

        q.put(
            {
                "ok": True,
                "times": times,
                "value": float(val),
                "peak_rss_bytes": int(peak_rss),
                "baseline_rss_bytes": int(baseline_rss),
                "contraction_rss_bytes": int(max(0, peak_rss - baseline_rss)),
            }
        )
    except Exception as e:  # noqa: BLE001
        # A memory blow-up may surface as a Python-level exception (MemoryError,
        # or torch/einsum "unable to allocate") before the OS OOM killer fires.
        # Classify these as oom so they read the same as a hard OOM kill.
        msg = repr(e).lower()
        is_oom = isinstance(e, MemoryError) or any(
            t in msg for t in ("out of memory", "unable to allocate",
                                "can't allocate", "cannot allocate", "bad_alloc")
        )
        q.put({"ok": False, "error": repr(e), "oom": bool(is_oom)})


def _run_with_timeout(target, args, timeout, mem_limit_bytes=None, poll=0.05):
    """Run `target(q, *args)` in a spawned process.

    Kills it if it runs past `timeout` seconds, or -- when `mem_limit_bytes` is
    set -- if its actual resident memory (RSS, including any children) exceeds
    the limit WHILE RUNNING. Memory is not pre-judged from an estimate: the
    contraction is launched and only aborted once real usage crosses the line.
    """
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    p = ctx.Process(target=target, args=(q, *args))
    p.start()

    proc = None
    if mem_limit_bytes:
        try:
            import psutil

            proc = psutil.Process(p.pid)
        except Exception:  # noqa: BLE001
            proc = None

    start = time.monotonic()
    peak_rss = 0
    while True:
        p.join(poll)
        if not p.is_alive():
            break
        if time.monotonic() - start > timeout:
            p.terminate()
            p.join()
            return {"ok": False, "error": "timeout", "timeout": True}
        if proc is not None:
            try:
                rss = proc.memory_info().rss
                for child in proc.children(recursive=True):
                    rss += child.memory_info().rss
                peak_rss = max(peak_rss, rss)
                if rss > mem_limit_bytes:
                    p.terminate()
                    p.join()
                    return {
                        "ok": False, "error": "mem", "oom": True,
                        "peak_rss_bytes": peak_rss,
                    }
            except Exception:  # noqa: BLE001 -- process may be tearing down
                pass
    try:
        return q.get_nowait()
    except Exception:  # noqa: BLE001 -- empty queue => worker died (e.g. OOM kill)
        return {"ok": False, "error": "no result (worker died -- OOM?)", "oom": True}


# ---------------------------------------------------------------------------
# Table rendering (one table per (m, n))
# ---------------------------------------------------------------------------


def _fmt_sci(v):
    return "n/a" if v is None else f"{v:.3e}"


def render_table(cell, scale_orders):
    """Build the per-(m, n) text table. One row per optimizer (no deps)."""
    scale_cols = [f"n^{s}" for s in scale_orders]
    columns = (
        ["optimizer", "path_time(s)", "exec_time(s)", "est_mem(elems)",
         "actual_mem(GB)", "flops", "scaling"]
        + scale_cols
        + ["u_value"]
    )

    records = []
    for r in cell["rows"]:
        rec = {
            "optimizer": r["optimizer"],
            "path_time(s)": (
                _fmt_sci(r["path_find_time_mean"])
                if r["path_ok"]
                else f"skipped({r['path_reason']})"
            ),
            "exec_time(s)": (
                _fmt_sci(r["exec_time_mean"])
                if r["exec_ok"]
                else f"skipped({r['exec_reason']})"
            ),
            "est_mem(elems)": _fmt_sci(r["peak_mem_elems"]) if r["path_ok"] else "-",
            "actual_mem(GB)": (
                f"{r['contraction_rss_bytes'] / 1e9:.3f}"
                if r.get("exec_ok") else "-"
            ),
            "flops": _fmt_sci(r["flops"]) if r["path_ok"] else "-",
            "scaling": str(r["scaling"]) if r["path_ok"] else "-",
        }
        hist = r.get("scaling_histogram", {})
        for s, col in zip(scale_orders, scale_cols):
            rec[col] = str(hist.get(f"n^{s}", 0)) if r["path_ok"] else "-"
        rec["u_value"] = _fmt_sci(r["u_value"]) if r["exec_ok"] else "-"
        records.append(rec)

    # column widths = max over header and cells
    widths = {
        c: max(len(c), *(len(rec[c]) for rec in records)) if records else len(c)
        for c in columns
    }
    sep = "  "
    head_line = sep.join(c.rjust(widths[c]) for c in columns)
    body_lines = [
        sep.join(rec[c].rjust(widths[c]) for c in columns) for rec in records
    ]
    table = "\n".join([head_line, *body_lines]) + "\n"

    header = (
        f"HOIF path-strategy report\n"
        f"Setting: m={cell['m']}, n={cell['n']}, dediag={cell['dediag']}, "
        f"repeats={cell['repeats']}, backend=torch/CPU, dtype={cell['dtype']}\n"
        f"Subexpressions (V-statistics) = Bell(m-1) = {cell['num_subexpressions']}\n"
        f"path_time = mean time to find contraction paths (all subexpressions).\n"
        f"exec_time = mean time to compute the U-statistic via UStats.compute.\n"
        f"est_mem / flops / scaling / n^s counts are static estimates of the "
        f"chosen paths (est_mem = largest intermediate, # elements).\n"
        f"actual_mem(GB) = measured peak process-RSS increase during the "
        f"contraction (real memory, excludes the resident input tensors).\n"
        f"'skipped(oom)'   = actual RSS exceeded --mem-limit-gb while running, "
        f"so the run was killed (est_mem is only informational).\n"
        f"'skipped(timeout)' = pass exceeded its wall-clock budget.\n"
        f"{'=' * 78}\n"
    )
    return header + table


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Per-(m, n) HOIF path-strategy report: estimated FLOPs / memory "
        "/ order distribution + path-finding time + actual execution time."
    )
    parser.add_argument(
        "--m", type=int, nargs="+", required=True, help="U-statistic orders m"
    )
    parser.add_argument(
        "--n", type=int, nargs="+", required=True, help="sample sizes n"
    )
    parser.add_argument(
        "--repeats", type=int, default=5, help="repeats for both timing passes"
    )
    parser.add_argument(
        "--optimizers",
        nargs="+",
        default=ALL_OPTIMIZERS,
        help=f"opt_einsum path optimizers (default: {' '.join(ALL_OPTIMIZERS)})",
    )
    parser.add_argument(
        "--mem-limit-gb",
        type=float,
        default=4.0,
        help="kill an execution if its ACTUAL RSS exceeds this many GB while "
        "running (not pre-judged from the estimate)",
    )
    parser.add_argument(
        "--path-timeout",
        type=float,
        default=600.0,
        help="wall-clock budget (s) for one optimizer's path-finding pass",
    )
    parser.add_argument(
        "--exec-timeout",
        type=float,
        default=600.0,
        help="wall-clock budget (s) for one optimizer's execution pass",
    )
    parser.add_argument(
        "--dtype", choices=["float64", "float32"], default="float32",
        help="execution precision; float32 matches the paper's run.py and is "
        "~2.6x faster than float64 at large n (default: float32)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--threads",
        type=int,
        default=int(_default_threads),
        help="CPU threads for torch/BLAS in the execution pass. Defaults to the "
        "SLURM allocation (SLURM_CPUS_PER_TASK) or os.cpu_count(). Set this to "
        "the cores your job actually reserved -- letting torch use all physical "
        "cores oversubscribes a partial allocation and is ~200x slower.",
    )
    parser.add_argument(
        "--skip-exec",
        action="store_true",
        help="only estimate / time path finding; do not run UStats.compute",
    )
    parser.add_argument("--dediag", action="store_true", default=True)
    parser.add_argument(
        "--no-dediag", dest="dediag", action="store_false",
        help="compute the V-statistic (no dediagonalization)",
    )
    parser.add_argument("--outdir", default=OUTPUT_DIR)
    args = parser.parse_args()

    # Re-pin threads to the requested count; spawned workers inherit these env
    # vars and also call torch.set_num_threads(args.threads) themselves.
    _pin_threads(args.threads)
    print(f"Using {args.threads} CPU thread(s) for execution.", flush=True)

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(args.outdir, f"path_complexity_full_{ts}.json")

    all_cells = []

    for m in args.m:
        for n in args.n:
            print(f"\n=== m={m}, n={n} (dediag={args.dediag}) ===", flush=True)
            rows = []
            num_sub = num_ops = None

            for opt in args.optimizers:
                # --- path finding pass ---
                pr = _run_with_timeout(
                    _path_worker, (m, n, opt, args.repeats, args.dediag),
                    args.path_timeout,
                )
                row = {"optimizer": opt}
                if pr.get("ok"):
                    times = pr["times"]
                    num_sub = pr["num_subexpressions"]
                    num_ops = pr["num_operands"]
                    row.update(
                        path_ok=True,
                        path_reason=None,
                        path_find_times=times,
                        path_find_time_mean=sum(times) / len(times),
                        scaling=pr["scaling"],
                        flops=pr["flops"],
                        peak_mem_elems=pr["largest"],
                        scaling_histogram={
                            f"n^{s}": pr["hist"][s] for s in sorted(pr["hist"])
                        },
                    )
                else:
                    reason = "timeout" if pr.get("timeout") else "error"
                    row.update(
                        path_ok=False, path_reason=reason,
                        path_error=pr.get("error"),
                        scaling=None, flops=None, peak_mem_elems=None,
                        scaling_histogram={},
                    )

                # --- execution pass ---
                # Memory is NOT pre-judged from the estimate: we launch the
                # contraction and only abort it if its real RSS crosses the
                # limit while running.
                if args.skip_exec:
                    row.update(exec_ok=False, exec_reason="disabled")
                elif not row["path_ok"]:
                    row.update(exec_ok=False, exec_reason="no-path")
                else:
                    er = _run_with_timeout(
                        _exec_worker,
                        (m, n, opt, args.repeats, args.dediag, args.seed,
                         args.dtype, args.threads),
                        args.exec_timeout,
                        mem_limit_bytes=int(args.mem_limit_gb * 1e9),
                    )
                    if er.get("ok"):
                        etimes = er["times"]
                        row.update(
                            exec_ok=True, exec_reason=None,
                            exec_times=etimes,
                            exec_time_mean=sum(etimes) / len(etimes),
                            u_value=er["value"],
                            peak_rss_bytes=er["peak_rss_bytes"],
                            contraction_rss_bytes=er["contraction_rss_bytes"],
                        )
                    else:
                        if er.get("timeout"):
                            reason = "timeout"
                        elif er.get("oom"):
                            reason = "oom"
                        else:
                            reason = "error"
                        row.update(
                            exec_ok=False, exec_reason=reason,
                            exec_error=er.get("error"),
                        )

                rows.append(row)

                # concise progress line
                pt = (
                    f"{row['path_find_time_mean']:.3e}s"
                    if row["path_ok"] else f"skip({row['path_reason']})"
                )
                xt = (
                    f"{row['exec_time_mean']:.3e}s"
                    if row.get("exec_ok") else f"skip({row['exec_reason']})"
                )
                fl = f"{row['flops']:.3e}" if row["path_ok"] else "-"
                mem = (
                    f"{row['contraction_rss_bytes'] / 1e9:.3f}GB"
                    if row.get("exec_ok") else "-"
                )
                print(
                    f"  {opt:<14} path={pt:<16} exec={xt:<18} "
                    f"flops={fl:<11} mem={mem:<9} scaling={row['scaling']}",
                    flush=True,
                )

            cell = {
                "m": m, "n": n, "dediag": args.dediag, "repeats": args.repeats,
                "dtype": args.dtype, "num_subexpressions": num_sub,
                "num_operands": num_ops, "rows": rows,
            }
            all_cells.append(cell)

            # per-(m, n) table -- written immediately so partial runs are saved
            scale_orders = sorted(
                {
                    int(k.split("^")[1])
                    for r in rows
                    for k in r.get("scaling_histogram", {})
                }
            )
            txt = render_table(cell, scale_orders)
            txt_path = os.path.join(
                args.outdir, f"path_complexity_m{m}_n{n}_{ts}.txt"
            )
            with open(txt_path, "w") as f:
                f.write(txt)
            print(f"  -> table saved to {txt_path}", flush=True)

    with open(json_path, "w") as f:
        json.dump(all_cells, f, indent=2, default=float)
    print(f"\nAll raw results saved to {json_path}")


if __name__ == "__main__":
    main()
