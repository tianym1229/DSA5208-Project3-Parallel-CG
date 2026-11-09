#!/usr/bin/env python3

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

CPU = ROOT / (
    "data/serial/serial_summary.csv"
)

GPU = ROOT / (
    "data/gpu/gpu_benchmark_summary.csv"
)

OUT = ROOT / (
    "results/analysis/gpu/cpu_gpu_comparison.csv"
)

TARGETS = {(2, 512), (3, 64)}

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

cpu_rows = read_csv(CPU)
gpu_rows = read_csv(GPU)

def index_rows(rows):
    result = {}
    for row in rows:
        key = (int(row["dim"]), int(row["grid"]))
        if key in result:
            raise RuntimeError(f"Duplicate configuration: {key}")
        result[key] = row
    return result

cpu = index_rows(cpu_rows)
gpu = index_rows(gpu_rows)

assert TARGETS <= cpu.keys()
assert TARGETS <= gpu.keys()

fields = [
    "dim",
    "grid",
    "n",
    "cpu_iterations",
    "gpu_min_iterations",
    "gpu_max_iterations",
    "cpu_repeats",
    "gpu_repeats",
    "cpu_median_solver_s",
    "gpu_median_solver_s",
    "gpu_median_end_to_end_s",
    "solver_time_ratio",
    "cpu_cv_percent",
    "gpu_solver_cv_percent",
    "cpu_high_variability",
]

output = []

for dim, grid in sorted(TARGETS):
    c = cpu[(dim, grid)]
    g = gpu[(dim, grid)]

    n = grid ** dim

    assert int(c["n"]) == n
    assert int(g["n"]) == n

    ci = int(c["iterations"])
    gi_min = int(g["min_iterations"])
    gi_max = int(g["max_iterations"])

    assert ci == gi_min == gi_max
    assert int(c["repeats"]) == 5
    assert int(g["repeats"]) == 5

    ct = float(c["median_s"])
    gt = float(g["median_solver_runtime_s"])
    ge = float(g["median_end_to_end_s"])

    assert all(
        math.isfinite(x) and x > 0
        for x in (ct, gt, ge)
    )

    row = {
        "dim": dim,
        "grid": grid,
        "n": n,
        "cpu_iterations": ci,
        "gpu_min_iterations": gi_min,
        "gpu_max_iterations": gi_max,
        "cpu_repeats": 5,
        "gpu_repeats": 5,
        "cpu_median_solver_s": ct,
        "gpu_median_solver_s": gt,
        "gpu_median_end_to_end_s": ge,
        "solver_time_ratio": ct / gt,
        "cpu_cv_percent": float(c["cv_percent"]),
        "gpu_solver_cv_percent": float(
            g["cv_solver_runtime_s_percent"]
        ),
        "cpu_high_variability": int(
            c["high_variability"]
        ),
    }

    output.append(row)

if OUT.exists():
    raise SystemExit(
        f"Refusing to overwrite existing file: {OUT}"
    )

with OUT.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(output)

print("===== CPU GPU COMPARISON =====")

for row in output:
    print(
        f"{row['dim']}D Grid {row['grid']}: "
        f"CPU={row['cpu_median_solver_s']:.6f}s, "
        f"GPU={row['gpu_median_solver_s']:.6f}s, "
        f"ratio={row['solver_time_ratio']:.3f}x"
    )

print("OUTPUT:", OUT)
print("===== COMPARISON COMPLETED =====")
