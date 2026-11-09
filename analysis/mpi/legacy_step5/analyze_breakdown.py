import csv
import statistics
from collections import defaultdict
from pathlib import Path

source = Path(
    "results/step5f_strong_scaling/results_1518104.csv"
)
output = Path(
    "results/step5k_breakdown/timing_breakdown.csv"
)

metrics = [
    "runtime_s",
    "halo_s",
    "spmv_s",
    "allreduce_s",
    "vector_s",
    "validation_s",
]

groups = defaultdict(list)

with source.open(newline="") as f:
    for row in csv.DictReader(f):
        key = (
            int(row["dim"]),
            int(row["grid"]),
            int(row["ranks"]),
            row["comm"],
        )
        groups[key].append(row)

results = []

for key in sorted(groups):
    dim, grid, ranks, comm = key
    rows = groups[key]

    result = {
        "dim": dim,
        "grid": grid,
        "ranks": ranks,
        "comm": comm,
    }

    for metric in metrics:
        values = [float(r[metric]) for r in rows]
        result[metric] = statistics.median(values)

    results.append(result)

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("===== TIMING BREAKDOWN =====")
print(
    f"{'Dim':<5} {'MPI':<5} {'Comm':<10} "
    f"{'Runtime':>10} {'Halo':>10} "
    f"{'SpMV':>10} {'Allreduce':>10}"
)

for r in results:
    print(
        f"{r['dim']:<5} "
        f"{r['ranks']:<5} "
        f"{r['comm']:<10} "
        f"{r['runtime_s']:>10.6f} "
        f"{r['halo_s']:>10.6f} "
        f"{r['spmv_s']:>10.6f} "
        f"{r['allreduce_s']:>10.6f}"
    )

print("\nSaved:", output)
print("STEP 5K TIMING BREAKDOWN COMPLETED")
