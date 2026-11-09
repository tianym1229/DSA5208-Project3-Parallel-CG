#!/usr/bin/env python3
import csv
import io
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build/cg_gpu"
import os

OUT = Path(
    os.environ.get(
        "CG_GPU_BENCHMARK_OUT",
        str(ROOT / "results/analysis/gpu/benchmark")
    )
)
OUT.mkdir(parents=True, exist_ok=True)

for filename in (
    "gpu_benchmark_raw.csv",
    "gpu_benchmark_summary.csv",
):
    if (OUT / filename).exists():
        raise SystemExit(
            f"Refusing to overwrite existing result: {OUT / filename}"
        )


TOL = 1e-8
REPEATS = 5
CASES = [(2, 512), (3, 64)]

HEADER = [
    "dim", "grid", "n", "nnz", "solution", "iterations",
    "solver_runtime_s", "h2d_s", "d2h_s",
    "validation_s", "end_to_end_s",
    "relative_residual", "relative_error", "converged"
]

TIMINGS = [
    "solver_runtime_s", "h2d_s", "d2h_s",
    "validation_s", "end_to_end_s"
]

raw_rows = []
summary_rows = []
failures = []

def run_case(dim, grid, phase, repeat):
    tag = f"{dim}d_grid{grid}_{phase}{repeat}"

    cmd = [
        str(EXE),
        "--dim", str(dim),
        "--grid", str(grid),
        "--solution", "benchmark",
        "--tol", str(TOL),
        "--max-iters", "10000",
    ]

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600
        )
    except subprocess.TimeoutExpired as exc:
        (OUT / f"{tag}.error.txt").write_text(str(exc))
        raise RuntimeError(f"{tag}: execution timeout") from exc

    (OUT / f"{tag}.stdout.txt").write_text(proc.stdout)
    (OUT / f"{tag}.stderr.txt").write_text(proc.stderr)

    if proc.returncode != 0:
        raise RuntimeError(
            f"{tag}: exit code {proc.returncode}; "
            f"see saved stdout/stderr"
        )

    lines = [s.strip() for s in proc.stdout.splitlines() if s.strip()]
    expected_header = ",".join(HEADER)

    if lines.count(expected_header) != 1:
        raise RuntimeError(f"{tag}: missing or duplicate CSV header")

    idx = lines.index(expected_header)
    if idx + 1 >= len(lines):
        raise RuntimeError(f"{tag}: missing CSV data row")

    parsed = list(csv.DictReader(
        io.StringIO("\n".join(lines[idx:idx+2]))
    ))

    if len(parsed) != 1:
        raise RuntimeError(f"{tag}: invalid CSV output")

    data = parsed[0]

    if (
        int(data["dim"]) != dim
        or int(data["grid"]) != grid
        or int(data["n"]) != grid**dim
        or data["solution"] != "benchmark"
    ):
        raise RuntimeError(f"{tag}: benchmark configuration mismatch")

    if int(data["converged"]) != 1:
        raise RuntimeError(f"{tag}: CG did not converge")

    iterations = int(data["iterations"])
    if not (0 <= iterations <= 10000):
        raise RuntimeError(f"{tag}: invalid iteration count")

    residual = float(data["relative_residual"])
    error = float(data["relative_error"])

    if not (math.isfinite(residual) and 0 <= residual < TOL):
        raise RuntimeError(f"{tag}: true residual validation failed")

    if not (math.isfinite(error) and error >= 0):
        raise RuntimeError(f"{tag}: invalid relative error")

    for field in TIMINGS:
        value = float(data[field])
        if not (math.isfinite(value) and value >= 0):
            raise RuntimeError(f"{tag}: invalid {field}")

    result = {
        "phase": phase,
        "repeat": repeat,
        **data
    }
    raw_rows.append(result)

    print(
        f"PASS {tag}: iterations={iterations}, "
        f"solver={float(data['solver_runtime_s']):.9f}s, "
        f"residual={residual:.3e}",
        flush=True
    )

def save_csv(path, rows, fields):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

print("===== STEP C7.3.5 GPU BENCHMARK =====", flush=True)

if not EXE.is_file():
    raise SystemExit(f"Missing executable: {EXE}")

for dim, grid in CASES:
    print(f"\n===== {dim}D GRID {grid} =====", flush=True)

    try:
        run_case(dim, grid, "warmup", 0)

        for repeat in range(1, REPEATS + 1):
            run_case(dim, grid, "measure", repeat)

        measured = [
            r for r in raw_rows
            if r["dim"] == str(dim)
            and r["grid"] == str(grid)
            and r["phase"] == "measure"
        ]

        if len(measured) != REPEATS:
            raise RuntimeError("Incomplete formal measurements")

        iterations = [int(r["iterations"]) for r in measured]

        summary = {
            "dim": dim,
            "grid": grid,
            "n": grid**dim,
            "repeats": REPEATS,
            "min_iterations": min(iterations),
            "max_iterations": max(iterations),
        }

        for field in TIMINGS:
            values = [float(r[field]) for r in measured]
            mean = statistics.mean(values)
            std = statistics.stdev(values)

            summary[f"median_{field}"] = statistics.median(values)
            summary[f"mean_{field}"] = mean
            summary[f"std_{field}"] = std
            summary[f"cv_{field}_percent"] = (
                100 * std / mean if mean > 0 else float("nan")
            )

        summary_rows.append(summary)

    except Exception as exc:
        message = f"{dim}D Grid {grid}: {exc}"
        failures.append(message)
        print("FAIL:", message, flush=True)

    finally:
        save_csv(
            OUT / "gpu_benchmark_raw.csv",
            raw_rows,
            ["phase", "repeat"] + HEADER
        )

if summary_rows:
    save_csv(
        OUT / "gpu_benchmark_summary.csv",
        summary_rows,
        list(summary_rows[0].keys())
    )

print("\n===== BENCHMARK SUMMARY =====")
print("Successful configurations:", len(summary_rows))
print("Failed configurations:", len(failures))
print("Output directory:", OUT)

for failure in failures:
    print("ERROR:", failure)

if failures or len(summary_rows) != len(CASES):
    sys.exit(1)

print("===== GPU BENCHMARK COMPLETED =====")
