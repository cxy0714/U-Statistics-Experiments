"""
motif_benchmark.py
==================
Unified benchmark for counting graph motifs using three methods:
  - ustat   : tensor-based U-statistics (fast, exact)
  - peregrine: compiled C++ counter (fast, exact)
  - igraph  : reference implementation (slow, used for verification)

Usage
-----
Edit the CONFIG block below, then run:
    python motif_benchmark.py

Mode switch
-----------
MODE = "fast"     -> ustat + Peregrine only  (parallelisable, low memory)
MODE = "igraph"   -> igraph only             (slow; give it dedicated cores)
MODE = "all"      -> all three methods       (+ optional consistency check)
"""

import os
import gc
import json
import time
import shutil
import tempfile
import subprocess
import concurrent.futures
from datetime import datetime

import numpy as np
import networkx as nx
import igraph as ig
import torch

from u_stats import ustat, set_backend
import argparse 
set_backend("torch")

# ---------------------------------------------------------------------------
# CONFIG – edit here
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Motif Benchmark")
parser.add_argument("--mode", type=str, default="fast", choices=["fast", "igraph", "all"], help="Running mode")
parser.add_argument("--verify", action="store_true", help="Enable consistency check")
args = parser.parse_args()

MODE = args.mode
VERIFY = args.verify or True  

# MODE = "igraph"          # "fast" | "igraph" | "all"
# VERIFY = True         # cross-check counts when all three results are present

CONFIGS = {
    3: {
        "n": 20_000,
        "p_values": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.4, 0.6, 0.8],
        "num_trials": 3,        # ≥3 recommended; median is reported
        "peregrine_threads": 40,
        "igraph_timeout": 30 * 60,   # seconds
    },
    4: {
        "n": 2_000,
        "p_values": [0.0005, 0.001, 0.005, 0.01, 0.05, 0.10, 0.20, 0.4, 0.6, 0.8],
        "num_trials": 3,
        "peregrine_threads": 40,
        "igraph_timeout": 60 * 60,
    },
}

OUTPUT_DIR = "experiments/motif_count/results"

# ---------------------------------------------------------------------------
# Peregrine paths
# ---------------------------------------------------------------------------

_SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
_PEREGRINE_DIR     = os.path.join(_PROJECT_ROOT, "Peregrine")
PEREGRINE_COUNT   = os.path.join(_PEREGRINE_DIR, "bin/count")
PEREGRINE_CONVERT = os.path.join(_PEREGRINE_DIR, "bin/convert_data")

# ---------------------------------------------------------------------------
# Motif metadata
# ---------------------------------------------------------------------------

# Peregrine pattern strings -> our motif IDs
PEREGRINE_3_MAPPING = {
    "[1-3](1~2)[2-3]": 1,   # V-shape (path)
    "[1-2][1-3][2-3]": 2,   # triangle
}

PEREGRINE_4_MAPPING = {
    "[1-2][1-3][1-4](2~3)(2~4)(3~4)": 1,
    "[1-2][1-4](1~3)[2-3](2~4)(3~4)": 2,
    "[1-2][1-3][1-4][2-3](2~4)(3~4)": 3,
    "[1-2][1-4](1~3)[2-3](2~4)[3-4]": 4,
    "[1-2][1-3][1-4][2-3](2~4)[3-4]": 5,
    "[1-2][1-3][1-4][2-3][2-4][3-4]": 6,
}

# Symmetry divisors for ustat raw counts
USTAT_DIVISORS_3 = {1: 2, 2: 6}
USTAT_DIVISORS_4 = {1: 6, 2: 2, 3: 2, 4: 8, 5: 4, 6: 24}

# igraph motifs_randesu index -> our motif IDs  (NaN entries are non-induced)
# igraph returns a list; indices with None are isomorphism classes that don't
# correspond to connected induced subgraphs we care about.
IGRAPH_3_MAP = {2: 1, 3: 2}          # igraph index -> our ID
IGRAPH_4_MAP = {4: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}

# ---------------------------------------------------------------------------
# ustat counting
# ---------------------------------------------------------------------------

def _ustat_count_3(A):
    t3 = [A, A, A]
    t2 = [A, A]
    m3 = [["i","j"], ["i","k"], ["j","k"]]
    m2 = [["i","j"], ["i","k"]]
    c_triangle = ustat(t3, m3, average=False)
    c_wedge_raw = ustat(t2, m2, average=False)
    return {
        1: int((c_wedge_raw - c_triangle) // USTAT_DIVISORS_3[1]),
        2: int(c_triangle // USTAT_DIVISORS_3[2]),
    }


def _ustat_count_4(A):
    d = USTAT_DIVISORS_4
    t6 = [A]*6;  t5 = [A]*5;  t4 = [A]*4;  t3 = [A]*3

    m6 = [["i","j"],["i","k"],["i","l"],["j","k"],["j","l"],["k","l"]]
    m5 = [["i","j"],["i","k"],["i","l"],["j","k"],["k","l"]]
    m4_a = [["i","j"],["i","l"],["j","k"],["k","l"]]
    m4_b = [["i","j"],["i","k"],["i","l"],["j","k"]]
    m3_a = [["i","j"],["i","l"],["j","k"]]
    m3_b = [["i","j"],["i","k"],["i","l"]]

    c6 = ustat(t6, m6, average=False)
    c5 = ustat(t5, m5, average=False)
    c4a = ustat(t4, m4_a, average=False)
    c4b = ustat(t4, m4_b, average=False)
    c3a = ustat(t3, m3_a, average=False)
    c3b = ustat(t3, m3_b, average=False)

    return {
        1: int((c3b - 3*c4b + 3*c5 - c6) // d[1]),
        2: int((c3a - 2*c4b - c4a + 3*c5 - c6) // d[2]),
        3: int((c4b - 2*c5 + c6) // d[3]),
        4: int((c4a - 2*c5 + c6) // d[4]),
        5: int((c5 - c6) // d[5]),
        6: int(c6 // d[6]),
    }


def count_ustat(G, size):
    A = nx.to_numpy_array(G)
    if size == 3:
        return _ustat_count_3(A)
    if size == 4:
        return _ustat_count_4(A)
    raise ValueError(f"Unsupported size: {size}")


def count_ustat_from_matrix(A, size):
    """Accept pre-computed adjacency matrix to avoid recomputing across trials."""
    if size == 3:
        return _ustat_count_3(A)
    if size == 4:
        return _ustat_count_4(A)
    raise ValueError(f"Unsupported size: {size}")

# ---------------------------------------------------------------------------
# Peregrine counting
# ---------------------------------------------------------------------------

def _save_edgelist(G, directory):
    path = os.path.join(directory, "edges.txt")
    with open(path, "w") as f:
        for u, v in G.edges():
            if u != v:
                f.write(f"{u+1} {v+1}\n")
    return path


def _convert_peregrine(edgelist_path, out_dir):
    r = subprocess.run(
        [PEREGRINE_CONVERT, edgelist_path, out_dir],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Peregrine convert failed:\n{r.stderr}")
    return out_dir


def _run_peregrine(graph_dir, size, threads):
    r = subprocess.run(
        [PEREGRINE_COUNT, graph_dir, f"{size}-motifs", str(threads)],
        capture_output=True, text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"Peregrine count failed:\n{r.stderr}")
    return r.stdout


def _parse_peregrine(stdout, size):
    mapping = PEREGRINE_3_MAPPING if size == 3 else PEREGRINE_4_MAPPING
    results = {}
    for line in stdout.strip().splitlines():
        if line.count(":") != 1:
            continue
        pat, cnt = line.split(":")
        pat = pat.strip()
        cnt = "".join(c for c in cnt if c.isdigit())
        if cnt and pat in mapping:
            results[mapping[pat]] = int(cnt)
    return results


def count_peregrine(G, size, threads=40):
    """Convert graph once, then run Peregrine count (only the count is timed)."""
    tmp = tempfile.mkdtemp(prefix="peregrine_")
    try:
        el = _save_edgelist(G, tmp)
        gd = _convert_peregrine(el, tmp)     # convert once, outside timed loop
        out = _run_peregrine(gd, size, threads)
        return _parse_peregrine(out, size)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def prepare_peregrine_graph(G):
    """
    Save and convert graph to Peregrine binary format.
    Returns (tmp_dir, graph_dir) – caller must clean up tmp_dir.
    This is done once per (G, size) so repeated timing runs skip the IO cost.
    """
    tmp = tempfile.mkdtemp(prefix="peregrine_")
    el  = _save_edgelist(G, tmp)
    gd  = _convert_peregrine(el, tmp)
    return tmp, gd


def count_peregrine_prepared(graph_dir, size, threads=40):
    """Run Peregrine on an already-converted graph directory."""
    out = _run_peregrine(graph_dir, size, threads)
    return _parse_peregrine(out, size)

# ---------------------------------------------------------------------------
# igraph counting  (runs in a subprocess with timeout)
# ---------------------------------------------------------------------------

def _igraph_worker(G_edges, n_nodes, size):
    """Top-level function so it can be pickled by ProcessPoolExecutor."""
    g = ig.Graph(n=n_nodes, edges=list(G_edges), directed=False)
    raw = g.motifs_randesu(size=size)
    mapping = IGRAPH_3_MAP if size == 3 else IGRAPH_4_MAP
    return {our_id: (raw[ig_idx] if ig_idx < len(raw) else None)
            for ig_idx, our_id in mapping.items()}


def _count_igraph_once(edges, n_nodes, size, timeout):
    """Run igraph in a subprocess once with a hard timeout.
    Returns (result_or_None, elapsed_seconds, error_or_None).
    """
    t0 = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_igraph_worker, edges, n_nodes, size)
        try:
            res = fut.result(timeout=timeout)
            return res, time.time() - t0, None
        except concurrent.futures.TimeoutError:
            fut.cancel()
            ex.shutdown(wait=False, cancel_futures=True)
            return None, time.time() - t0, "timeout"
        except Exception as e:
            return None, time.time() - t0, str(e)


def count_igraph_median(G, size, timeout=1800, trials=3):
    """Run igraph multiple times and return the result at median elapsed time.
    The graph is serialised to edges/n_nodes once; each trial just re-runs
    the subprocess.  If any trial times out or errors, the whole call aborts
    immediately – no point retrying a graph that already hit the limit.
    Returns (result_or_None, elapsed_seconds, error_or_None).
    """
    edges   = list(G.edges())
    n_nodes = G.number_of_nodes()
    records = []
    for _ in range(trials):
        r, t, e = _count_igraph_once(edges, n_nodes, size, timeout)
        if e:
            return None, t, e
        records.append((t, r))
    records.sort(key=lambda x: x[0])
    t_med, r_med = records[len(records) // 2]
    return r_med, t_med, None

# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def timed(fn, *args, **kwargs):
    """Return (result, elapsed_seconds, error_or_None)."""
    try:
        gc.collect()
        t0 = time.time()
        res = fn(*args, **kwargs)
        return res, time.time() - t0, None
    except Exception as e:
        return None, time.time() - t0, str(e)


def median_timed(fn, *args, trials=3, **kwargs):
    """Run fn multiple times and return result/time at median elapsed."""
    records = []
    for _ in range(trials):
        r, t, e = timed(fn, *args, **kwargs)
        if e:
            return None, None, e
        records.append((t, r))
    records.sort(key=lambda x: x[0])
    t_med, r_med = records[len(records) // 2]
    return r_med, t_med, None

# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def verify_counts(ustat_res, peregrine_res, igraph_res):
    """
    Compare counts across methods. Returns a dict of {motif_id: status}.
    status is "ok", "mismatch", or "missing".
    """
    all_ids = set()
    for r in (ustat_res, peregrine_res, igraph_res):
        if r:
            all_ids |= set(r.keys())

    report = {}
    for mid in sorted(all_ids):
        vals = {
            name: res.get(mid)
            for name, res in [("ustat", ustat_res), ("peregrine", peregrine_res), ("igraph", igraph_res)]
            if res is not None
        }
        non_null = [v for v in vals.values() if v is not None]
        if len(non_null) < 2:
            report[mid] = {"status": "missing", "values": vals}
        elif len(set(non_null)) == 1:
            report[mid] = {"status": "ok", "values": vals}
        else:
            report[mid] = {"status": "mismatch", "values": vals}
    return report

# ---------------------------------------------------------------------------
# Single experiment
# ---------------------------------------------------------------------------

def run_experiment(G, size, cfg, mode, verify):
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    trials  = cfg["num_trials"]
    threads = cfg["peregrine_threads"]
    timeout = cfg["igraph_timeout"]

    rec = {
        "graph": {"nodes": n_nodes, "edges": n_edges},
        "ustat":     {"result": None, "time": None, "error": None},
        "peregrine": {"result": None, "time": None, "error": None},
        "igraph":    {"result": None, "time": None, "error": None},
        "verify":    None,
    }

    if mode in ("fast", "all"):
        # Pre-compute adjacency matrix once; only the tensor contractions are timed.
        A = nx.to_numpy_array(G)
        print(f"    [ustat] ...", end=" ", flush=True)
        r, t, e = median_timed(count_ustat_from_matrix, A, size, trials=trials)
        rec["ustat"] = {"result": r, "time": t, "error": e}
        print(f"{t:.3f}s" if t else f"ERROR: {e}")

        # Convert graph to Peregrine binary format once; only the count subprocess is timed.
        peregrine_tmp = None
        try:
            print(f"    [peregrine] converting graph ...", end=" ", flush=True)
            peregrine_tmp, graph_dir = prepare_peregrine_graph(G)
            print("done")
            print(f"    [peregrine] counting ...", end=" ", flush=True)
            r, t, e = median_timed(count_peregrine_prepared, graph_dir, size, threads, trials=trials)
            rec["peregrine"] = {"result": r, "time": t, "error": e}
            print(f"{t:.3f}s" if t else f"ERROR: {e}")
        except Exception as e:
            rec["peregrine"] = {"result": None, "time": None, "error": str(e)}
            print(f"ERROR: {e}")
        finally:
            if peregrine_tmp:
                shutil.rmtree(peregrine_tmp, ignore_errors=True)

    if mode in ("igraph", "all"):
        print(f"    [igraph] {trials} trial(s), timeout={timeout}s each ...", end=" ", flush=True)
        r, t, e = count_igraph_median(G, size, timeout=timeout, trials=trials)
        rec["igraph"] = {"result": r, "time": t, "error": e}
        print(f"{t:.3f}s" if t else f"ERROR: {e}")

    if verify and mode == "all":
        rec["verify"] = verify_counts(
            rec["ustat"]["result"],
            rec["peregrine"]["result"],
            rec["igraph"]["result"],
        )
        mismatches = [k for k, v in rec["verify"].items() if v["status"] == "mismatch"]
        if mismatches:
            print(f"    [verify] MISMATCH on motif IDs: {mismatches}")
        else:
            print(f"    [verify] all counts consistent")

    return rec

# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(size, mode=MODE, verify=VERIFY):
    cfg = CONFIGS[size]
    n         = cfg["n"]
    p_values  = cfg["p_values"]
    trials    = cfg["num_trials"]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"benchmark_size{size}_{mode}_{ts}.json")

    payload = {
        "meta": {
            "motif_size": size,
            "mode": mode,
            "verify": verify,
            "n": n,
            "p_values": p_values,
            "num_trials": trials,
            "peregrine_threads": cfg["peregrine_threads"],
            "igraph_timeout": cfg["igraph_timeout"],
            "started_at": datetime.now().isoformat(),
        },
        "experiments": [],
    }

    total = len(p_values) * trials
    done  = 0
    igraph_timed_out = False  # <-- early-stop flag
    
    print(f"\n{'='*65}")
    print(f"  Motif size={size}  mode={mode}  n={n}  trials={trials}")
    print(f"{'='*65}")

    for p in p_values:
        if igraph_timed_out and mode in ("igraph", "all"):
            print(f"\n  [skip] p={p:.3f} — igraph already timed out, skipping remaining p values")
            break
        
        for trial in range(trials):
            seed = hash((n, size, p, trial)) & 0xFFFF_FFFF
            G = nx.gnp_random_graph(n, p, seed=seed)
            G = nx.convert_node_labels_to_integers(G)
            G.remove_nodes_from(list(nx.isolates(G)))

            done += 1
            print(f"\n  [{done}/{total}] p={p:.2f}  trial={trial}  "
                  f"nodes={G.number_of_nodes()}  edges={G.number_of_edges()}")

            rec = run_experiment(G, size, cfg, mode, verify)
            rec.update({"size": size, "n": n, "p": p, "trial": trial,
                        "seed": seed, "timestamp": datetime.now().isoformat()})
            payload["experiments"].append(rec)
            
            # Check for igraph timeout and set early-stop flag
            if rec["igraph"]["error"] == "timeout":
                igraph_timed_out = True
                print(f"    [early stop] igraph timed out — will skip remaining p values")

            # Save after every experiment so nothing is lost on crash
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)

            if igraph_timed_out and mode in ("igraph", "all"):
                break  # break inner trial loop too
            
            # Save after every experiment so nothing is lost on crash
            with open(out_path, "w") as f:
                json.dump(payload, f, indent=2, default=str)

    print(f"\n  Saved -> {out_path}")
    return payload

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Sanity-check Peregrine binaries when they'll be used
    if MODE in ("fast", "all"):
        for bin_path in (PEREGRINE_COUNT, PEREGRINE_CONVERT):
            if not os.path.exists(bin_path):
                raise FileNotFoundError(f"Peregrine binary not found: {bin_path}")

    print(f"Starting benchmark  |  mode={MODE}  |  {datetime.now():%Y-%m-%d %H:%M:%S}")

    run_benchmark(size=3, mode=MODE, verify=VERIFY)
    run_benchmark(size=4, mode=MODE, verify=VERIFY)

    print(f"\nAll done.  {datetime.now():%Y-%m-%d %H:%M:%S}")


if __name__ == "__main__":
    main()