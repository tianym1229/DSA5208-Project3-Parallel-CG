import csv
import statistics
from collections import defaultdict
from pathlib import Path

source = Path(
    "results/step5f_strong_scaling/results_1518104.csv"
)

output = Path(
    "results/step5h_analysis/scaling_summary.csv"
)

groups = defaultdict(list)

with source.open(newline="") as f:
    for row in csv.DictReader(f):
        key = (
            int(row["dim"]),
            int(row["grid"]),
            int(row["ranks"]),
            row["comm"],
        )
        groups[key].append(float(row["runtime_s"]))

medians = {
    key: statistics.median(values)
    for key, values in groups.items()
}

summary = []

for key in sorted(groups):
    dim, grid, ranks, comm = key
    values = groups[key]

    median = statistics.median(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)

    baseline_one = medians[(dim, grid, 1, comm)]

    speedup = baseline_one / median
    efficiency = speedup / ranks

    baseline_same = medians[(dim, grid, ranks, "baseline")]

    overlap_improvement = (
        (baseline_same - median) / baseline_same * 100
    )

    summary.append({
        "dim": dim,
        "grid": grid,
        "ranks": ranks,
        "comm": comm,
        "repeats": len(values),
        "median_s": median,
        "mean_s": mean,
        "std_s": std,
        "speedup": speedup,
        "efficiency": efficiency,
        "overlap_improvement_pct": overlap_improvement,
    })

fields = list(summary[0].keys())

with output.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print("===== STRONG SCALING SUMMARY =====")
print(
    f"{'Dim':<5} {'Grid':<6} {'MPI':<5} "
    f"{'Comm':<10} {'Median(s)':>12} "
    f"{'Speedup':>10} {'Efficiency':>12}"
)

for row in summary:
    print(
        f"{row['dim']:<5} "
        f"{row['grid']:<6} "
        f"{row['ranks']:<5} "
        f"{row['comm']:<10} "
        f"{row['median_s']:>12.6f} "
        f"{row['speedup']:>10.3f} "
        f"{row['efficiency']:>12.3f}"
    )

print("\nSaved:", output)
