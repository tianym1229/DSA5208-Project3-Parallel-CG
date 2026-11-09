import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BASE = Path("data/serial")
OUT = Path("results/analysis/serial/figures")
OUT.mkdir(parents=True, exist_ok=True)

with (BASE / "serial_summary.csv").open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 10, "Expected 10 configurations"

data = {}

for dim in (2, 3):
    subset = sorted(
        [r for r in rows if int(r["dim"]) == dim],
        key=lambda r: int(r["n"])
    )

    assert len(subset) == 5

    n = np.array([int(r["n"]) for r in subset], dtype=float)
    runtime = np.array(
        [float(r["median_s"]) for r in subset], dtype=float
    )
    iterations = np.array(
        [int(r["iterations"]) for r in subset], dtype=float
    )
    grids = [int(r["grid"]) for r in subset]

    assert np.all(n > 0)
    assert np.all(runtime > 0)
    assert np.all(iterations > 0)

    data[dim] = {
        "n": n,
        "runtime": runtime,
        "iterations": iterations,
        "grids": grids,
        "time_per_iter_ms": 1000 * runtime / iterations,
    }

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def save(fig, name):
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print("CREATED:", path)
    plt.close(fig)

# Figure S1: Runtime log-log scaling and power-law fit

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, dim in zip(axes, (2, 3)):
    d = data[dim]
    n = d["n"]
    t = d["runtime"]

    alpha, beta = np.polyfit(np.log(n), np.log(t), 1)
    fitted_log = alpha * np.log(n) + beta
    ss_res = np.sum((np.log(t) - fitted_log)**2)
    ss_tot = np.sum((np.log(t) - np.mean(np.log(t)))**2)
    r2 = 1 - ss_res / ss_tot

    xfit = np.geomspace(n.min(), n.max(), 200)
    yfit = np.exp(beta) * xfit**alpha

    ax.loglog(
        n, t, "o-", linewidth=1.8,
        markersize=6, label="Measured median"
    )
    ax.loglog(
        xfit, yfit, "--", linewidth=1.8,
        label=f"Power-law fit: alpha={alpha:.3f}"
    )

    ax.set_title(f"{dim}D Serial CG")
    ax.set_xlabel("Number of unknowns (N)")
    ax.set_ylabel("Median runtime (s)")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(loc="best")

    ax.text(
        0.04, 0.05,
        f"Log-space R² = {r2:.4f}",
        transform=ax.transAxes,
        fontsize=9,
        bbox=dict(
            facecolor="white",
            edgecolor="0.8",
            alpha=0.9
        )
    )

    print(
        f"FIT {dim}D: alpha={alpha:.6f}, "
        f"beta={beta:.6f}, R2={r2:.6f}"
    )

fig.suptitle("Serial CG Runtime Scaling", fontsize=14)
fig.tight_layout()
save(fig, "figure_S1_runtime_loglog")

# Figure S2: CG iteration scaling

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, dim in zip(axes, (2, 3)):
    d = data[dim]

    ax.plot(
        d["n"], d["iterations"],
        "o-", linewidth=1.8, markersize=6
    )

    ax.set_xscale("log")
    ax.set_title(f"{dim}D Serial CG")
    ax.set_xlabel("Number of unknowns (N)")
    ax.set_ylabel("CG iterations")
    ax.grid(True, which="both", alpha=0.25)

fig.suptitle("Serial CG Iteration Scaling", fontsize=14)
fig.tight_layout()
save(fig, "figure_S2_iterations")

# Figure S3: Time per iteration

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

for ax, dim in zip(axes, (2, 3)):
    d = data[dim]

    ax.loglog(
        d["n"], d["time_per_iter_ms"],
        "o-", linewidth=1.8, markersize=6
    )

    ax.set_title(f"{dim}D Serial CG")
    ax.set_xlabel("Number of unknowns (N)")
    ax.set_ylabel("Average time per iteration (ms)")
    ax.grid(True, which="both", alpha=0.25)

fig.suptitle("Serial CG Time per Iteration", fontsize=14)
fig.tight_layout()
save(fig, "figure_S3_time_per_iteration")

print()
print("===== FINAL STATUS =====")
print("SERIAL FINAL FIGURES GENERATED")
