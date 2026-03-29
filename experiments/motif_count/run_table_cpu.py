import json
from collections import defaultdict

FILE_PAIRS = [
    (
        "experiments/motif_count/results/benchmark_size4_fast_20260328_173329.json",
        "experiments/motif_count/results/benchmark_size4_igraph_20260328_155723.json",
    ),
    (
        "experiments/motif_count/results/benchmark_size3_fast_20260328_143440.json",
        "experiments/motif_count/results/benchmark_size3_igraph_20260328_143404.json",
    ),
]

OUTPUT_DIR = "experiments/motif_count/results"


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def average(lst):
    return sum(lst) / len(lst) if lst else None


def get_motif_size(data):
    return data.get("meta", {}).get("motif_size", "?")


def process_fast(data):
    grouped = defaultdict(lambda: {"ustat": [], "peregrine": []})

    for exp in data["experiments"]:
        p = exp["p"]
        if exp["ustat"]["time"] is not None:
            grouped[p]["ustat"].append(exp["ustat"]["time"])
        if exp["peregrine"]["time"] is not None:
            grouped[p]["peregrine"].append(exp["peregrine"]["time"])

    return {
        p: (average(grouped[p]["ustat"]), average(grouped[p]["peregrine"]))
        for p in grouped
    }


def process_igraph(data):
    grouped = defaultdict(list)
    timeout_ps = set()

    for exp in data["experiments"]:
        p = exp["p"]
        t = exp["igraph"]["time"]
        err = exp["igraph"]["error"]

        if err == "timeout":
            timeout_ps.add(p)
        elif t is not None:
            grouped[p].append(t)

    result = {}
    seen_timeout = False
    for p in sorted(set(list(grouped.keys()) + list(timeout_ps))):
        if p in timeout_ps:
            seen_timeout = True
            result[p] = "OOT"
        elif seen_timeout:
            result[p] = "OOT"
        else:
            result[p] = average(grouped[p])

    return result


def format_float(x):
    return f"{x:.4f}"


def build_table(fast_data, igraph_data):
    motif_size = get_motif_size(fast_data)
    fast = process_fast(fast_data)
    igraph = process_igraph(igraph_data)
    ps = sorted(fast.keys())

    lines = []
    lines.append(r"\begin{tabular}{cccc}")
    lines.append(r"\toprule")
    lines.append(
        r"\makecell{\textbf{Edge} \\ \textbf{Prob.} $p$} & "
        r"\makecell{\textbf{ustat} \\ \textbf{Time (s)}} & "
        r"\makecell{\textbf{Peregrine} \\ \textbf{Time (s)}} & "
        r"\makecell{\textbf{igraph} \\ \textbf{Time (s)}} \\"
    )
    lines.append(r"\midrule")

    for p in ps:
        ustat_time, peregrine_time = fast[p]
        ig_time = igraph.get(p, None)

        ig_str = "OOT" if (ig_time == "OOT" or ig_time is None) else format_float(ig_time)

        lines.append(
            f"{p:.4f} & "
            f"{format_float(ustat_time)} & "
            f"{format_float(peregrine_time)} & "
            f"{ig_str} \\\\"
        )

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")

    return motif_size, "\n".join(lines)


def main():
    for fast_file, igraph_file in FILE_PAIRS:
        fast_data = load_json(fast_file)
        igraph_data = load_json(igraph_file)

        motif_size, table = build_table(fast_data, igraph_data)

        out_path = f"{OUTPUT_DIR}/table_size{motif_size}.txt"
        with open(out_path, "w") as f:
            f.write(table)

        print(f"[motif_size={motif_size}] 已保存到 {out_path}")
        print(table)
        print()


if __name__ == "__main__":
    main()