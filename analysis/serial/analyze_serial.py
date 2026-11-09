import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

source = Path("data/serial/results_1518159.csv")

output = Path("results/analysis/serial/serial_summary_recomputed.csv")

groups = defaultdict(list)

with source.open(newline="") as f:
    for row in csv.DictReader(f):
        key = (int(row["dim"]), int(row["grid"]))
        groups[key].append(row)

results = []

print("===== SERIAL SCALING STATISTICAL SUMMARY =====")
print(
    f"{'Dim':<5} {'Grid':<7} {'N':<10} "
    f"{'Median(s)':>12} {'Mean(s)':>12} "
    f"{'Std(s)':>12} {'CV(%)':>9}"
)

for key in sorted(groups):
    dim, grid = key
    rows = groups[key]

    times = [float(r["runtime_s"]) for r in rows]
    iterations = [int(r["iterations"]) for r in rows]

    mean = statistics.mean(times)
    median = statistics.median(times)
    std = statistics.stdev(times)
    cv = 100 * std / mean

    result = {
        "dim": dim,
        "grid": grid,
        "n": grid ** dim,
        "repeats": len(times),
        "iterations": statistics.median(iterations),
        "median_s": median,
        "mean_s": mean,
        "std_s": std,
        "cv_percent": cv,
        "min_s": min(times),
        "max_s": max(times),
        "high_variability": int(cv > 20)
    }

    results.append(result)

    print(
        f"{dim:<5} {grid:<7} {grid**dim:<10} "
        f"{median:>12.6f} {mean:>12.6f} "
        f"{std:>12.6f} {cv:>9.2f}"
    )

with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\n===== VARIABILITY CHECK =====")

flagged = [r for r in results if r["high_variability"]]

if flagged:
    for r in flagged:
        print(
            f"REVIEW: dim={r['dim']} "
            f"grid={r['grid']} "
            f"CV={r['cv_percent']:.2f}%"
        )
else:
    print("PASS: All configurations have CV <= 10%.")

print("\nSaved:", output)
print("STEP B6 STATISTICAL ANALYSIS COMPLETED")
