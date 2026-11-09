#!/usr/bin/env python3
import csv
import io
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXE = ROOT / "build/cg_gpu"
import os

OUT = Path(
    os.environ.get(
        "CG_GPU_CORRECTNESS_OUT",
        str(ROOT / "results/analysis/gpu/correctness")
    )
)
OUT.mkdir(parents=True, exist_ok=True)

TOL = 1e-8

CASES = [
    ("G1", 2, 8),
    ("G2", 3, 8),
    ("G3", 2, 32),
    ("G4", 3, 16),
]

FIELDS = [
    "test", "dim", "grid", "n",
    "iterations", "solver_runtime_s",
    "relative_residual", "relative_error",
    "converged", "exit_code", "status", "reason"
]

results = []

print("===== STEP C7.3.3 GPU CORRECTNESS TEST =====", flush=True)

if not EXE.is_file():
    raise SystemExit(f"Missing executable: {EXE}")

for name, dim, grid in CASES:
    print(f"\n===== {name}: {dim}D GRID {grid} =====", flush=True)

    cmd = [
        str(EXE),
        "--dim", str(dim),
        "--grid", str(grid),
        "--solution", "benchmark",
        "--tol", str(TOL),
        "--max-iters", "10000",
    ]

    row = {field: "" for field in FIELDS}
    row.update(test=name, dim=dim, grid=grid, n=grid**dim)

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=180
        )

        row["exit_code"] = proc.returncode

        (OUT / f"{name}.stdout.txt").write_text(proc.stdout)
        (OUT / f"{name}.stderr.txt").write_text(proc.stderr)

        lines = [
            line.strip()
            for line in proc.stdout.splitlines()
            if line.strip()
        ]

        header = (
            "dim,grid,n,nnz,solution,iterations,"
            "solver_runtime_s,h2d_s,d2h_s,"
            "validation_s,end_to_end_s,"
            "relative_residual,relative_error,converged"
        )

        index = lines.index(header)

        reader = csv.DictReader(
            io.StringIO("\n".join(lines[index:index+2]))
        )

        data = next(reader)

        row["iterations"] = int(data["iterations"])
        row["solver_runtime_s"] = float(data["solver_runtime_s"])
        row["relative_residual"] = float(data["relative_residual"])
        row["relative_error"] = float(data["relative_error"])
        row["converged"] = int(data["converged"])

        checks = [
            (proc.returncode == 0, "nonzero exit code"),
            (int(data["dim"]) == dim, "dimension mismatch"),
            (int(data["grid"]) == grid, "grid mismatch"),
            (int(data["n"]) == grid**dim, "problem size mismatch"),
            (data["solution"] == "benchmark", "solution mismatch"),
            (row["converged"] == 1, "not converged"),
            (
                math.isfinite(row["relative_residual"])
                and row["relative_residual"] < TOL,
                "true residual failed"
            ),
            (
                math.isfinite(row["relative_error"])
                and row["relative_error"] >= 0,
                "invalid relative error"
            ),
            (
                math.isfinite(row["solver_runtime_s"])
                and row["solver_runtime_s"] >= 0,
                "invalid solver runtime"
            ),
            (
                0 <= row["iterations"] <= 10000,
                "invalid iteration count"
            ),
        ]

        failures = [
            reason for passed, reason in checks if not passed
        ]

        row["status"] = "PASS" if not failures else "FAIL"
        row["reason"] = "; ".join(failures)

    except Exception as exc:
        row["status"] = "FAIL"
        row["reason"] = str(exc)

    results.append(row)

    print(
        f"{name}: {row['status']} "
        f"iterations={row['iterations']} "
        f"residual={row['relative_residual']} "
        f"error={row['relative_error']}",
        flush=True
    )

    if row["reason"]:
        print("REASON:", row["reason"], flush=True)

summary = OUT / "gpu_correctness_summary.csv"

with summary.open("x", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(results)

passed = sum(row["status"] == "PASS" for row in results)

print("\n===== GPU CORRECTNESS SUMMARY =====")
print(f"PASSED: {passed}/{len(CASES)}")
print(f"SUMMARY: {summary}")

if passed != len(CASES):
    print("===== GPU CORRECTNESS TEST FAILED =====")
    sys.exit(1)

print("===== GPU CORRECTNESS TEST PASSED =====")
