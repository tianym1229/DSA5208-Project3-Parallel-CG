import csv
import math
import statistics
from pathlib import Path
from collections import defaultdict

FILES = {
    4: "data/hybrid/formal_measurements_1526943.csv",
    8: "data/hybrid/formal_measurements_1526965.csv",
    16: "data/hybrid/formal_measurements_1526970.csv",
    32: "data/hybrid/formal_measurements_1526980.csv",
}

OUTDIR = Path("results/analysis/hybrid")
OUTDIR.mkdir(parents=True, exist_ok=True)

OUTPUT = OUTDIR / "hybrid_formal_summary.csv"

groups = defaultdict(list)

for cores, filename in FILES.items():
    with open(filename, newline="") as f:
        for row in csv.DictReader(f):
            if row["phase"] != "measure":
                continue

            key = (
                cores,
                int(row["dim"]),
                int(row["grid"]),
                int(row["ranks"]),
                int(row["omp_threads"]),
                row["comm"],
            )
            groups[key].append(row)

metrics = [
    "spmv_s",
    "halo_s",
    "allreduce_s",
    "vector_s",
    "relative_residual",
    "relative_error",
]

summary = []

for key in sorted(groups):
    cores, dim, grid, ranks, threads, comm = key
    data = groups[key]

    assert len(data) == 5, f"Invalid repeats: {key}"
    assert sorted(int(r["repeat"]) for r in data) == [1, 2, 3, 4, 5]
    assert all(int(r["converged"]) == 1 for r in data)

    times = [float(r["runtime_s"]) for r in data]
    iterations = {int(r["iterations"]) for r in data}

    assert len(iterations) == 1
    iters = next(iter(iterations))

    mean = statistics.mean(times)
    median = statistics.median(times)
    sd = statistics.stdev(times)
    cv = 100 * sd / mean

    result = {
        "cores": cores,
        "dim": dim,
        "grid": grid,
        "n": grid ** dim,
        "ranks": ranks,
        "threads": threads,
        "comm": comm,
        "repeats": len(data),
        "iterations": iters,
        "median_runtime_s": median,
        "mean_runtime_s": mean,
        "std_runtime_s": sd,
        "cv_percent": cv,
        "min_runtime_s": min(times),
        "max_runtime_s": max(times),
        "time_per_iteration_ms": 1000 * median / iters,
    }

    for metric in metrics:
        values = [float(r[metric]) for r in data]
        assert all(math.isfinite(v) and v >= 0 for v in values)
        result[f"median_{metric}"] = statistics.median(values)

    summary.append(result)

assert len(summary) == 48, "Expected 48 configurations"

fields = list(summary[0].keys())

if OUTPUT.exists():
    raise SystemExit(f"Refusing to overwrite: {OUTPUT}")

with OUTPUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(summary)

print("===== STEP C6.2.6.2 UNIFIED STATISTICAL SUMMARY =====")
print("Configurations:", len(summary))
print("Formal measurements:", sum(int(r["repeats"]) for r in summary))
print("Output:", OUTPUT)

print("\n===== MEDIAN RUNTIME SUMMARY =====")
print(
    f"{'CORES':>5} {'DIM':>4} {'COMM':<9} "
    f"{'CONFIG':>7} {'MEDIAN(s)':>11} {'CV%':>8}"
)

for r in summary:
    print(
        f"{r['cores']:>5} "
        f"{r['dim']:>4} "
        f"{r['comm']:<9} "
        f"{str(r['ranks'])+'x'+str(r['threads']):>7} "
        f"{r['median_runtime_s']:>11.6f} "
        f"{r['cv_percent']:>8.2f}"
    )

high_cv = [r for r in summary if r["cv_percent"] > 10]

print("\n===== HIGH VARIABILITY =====")
print("High-CV configurations:", len(high_cv))

for r in high_cv:
    print(
        f"CORES={r['cores']} DIM={r['dim']} "
        f"CONFIG={r['ranks']}x{r['threads']} "
        f"COMM={r['comm']} CV={r['cv_percent']:.2f}%"
    )

print("\n===== FINAL STATUS =====")
print("UNIFIED HYBRID STATISTICAL SUMMARY COMPLETED")
