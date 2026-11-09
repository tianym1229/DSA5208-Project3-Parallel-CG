#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

COMPARISON = ROOT / (
    "results/analysis/gpu/cpu_gpu_comparison.csv"
)

GPU_SUMMARY = ROOT / (
    "data/gpu/gpu_benchmark_summary.csv"
)

OUT = ROOT / "results/analysis/gpu/figures"
OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

comparison = read_csv(COMPARISON)
gpu_summary = read_csv(GPU_SUMMARY)

comparison.sort(key=lambda r: int(r["dim"]))
gpu_summary.sort(key=lambda r: int(r["dim"]))

assert len(comparison) == 2
assert len(gpu_summary) == 2

labels = [
    f"{r['dim']}D Grid {r['grid']}"
    for r in comparison
]

cpu = np.array([
    float(r["cpu_median_solver_s"])
    for r in comparison
])

gpu = np.array([
    float(r["gpu_median_solver_s"])
    for r in comparison
])

ratio = np.array([
    float(r["solver_time_ratio"])
    for r in comparison
])

x = np.arange(len(labels))
width = 0.34

plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.dpi": 150,
    "savefig.dpi": 300,
})

def save_figure(fig, name):
    for ext in ("png", "pdf"):
        path = OUT / f"{name}.{ext}"
        if path.exists():
            raise FileExistsError(
                f"Refusing to overwrite: {path}"
            )
        fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

# G1: CPU vs GPU solver runtime
fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x - width/2, cpu, width, label="Serial CPU")
ax.bar(x + width/2, gpu, width, label="NVIDIA A40 GPU")

ax.set_yscale("log")
ax.set_xticks(x, labels)
ax.set_ylabel("Median solver runtime (s, log scale)")
ax.set_title("CPU vs GPU Solver Runtime")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
save_figure(fig, "figure_G1_cpu_gpu_runtime")

# G2: Observed CPU/GPU solver-time ratio
fig, ax = plt.subplots(figsize=(8, 5))

bars = ax.bar(x, ratio, width=0.5)

ax.set_xticks(x, labels)
ax.set_ylabel("CPU / GPU median solver-time ratio")
ax.set_title("Observed CPU/GPU Solver-Time Ratio")
ax.set_ylim(0, max(ratio) * 1.25)
ax.grid(axis="y", alpha=0.25)

for bar, value in zip(bars, ratio):
    ax.text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height(),
        f"{value:.2f}x",
        ha="center",
        va="bottom"
    )

fig.tight_layout()
save_figure(fig, "figure_G2_solver_time_ratio")

# G3: GPU runtime components
fields = [
    ("Solver", "median_solver_runtime_s"),
    ("H2D", "median_h2d_s"),
    ("D2H", "median_d2h_s"),
    ("Validation", "median_validation_s"),
    ("End-to-End", "median_end_to_end_s"),
]

fig, ax = plt.subplots(figsize=(10, 5.5))

component_labels = [name for name, _ in fields]
positions = np.arange(len(fields))

bar_width = 0.35

for i, row in enumerate(gpu_summary):
    values = [
        1000 * float(row[field])
        for _, field in fields
    ]

    ax.bar(
        positions + (i - 0.5) * bar_width,
        values,
        bar_width,
        label=f"{row['dim']}D Grid {row['grid']}"
    )

ax.set_xticks(positions, component_labels)
ax.set_ylabel("Median runtime (ms)")
ax.set_title("GPU Runtime Metrics")
ax.legend()
ax.grid(axis="y", alpha=0.25)

fig.tight_layout()
save_figure(fig, "figure_G3_gpu_runtime_components")

print("===== GPU FIGURES GENERATED =====")

for path in sorted(OUT.iterdir()):
    print(path)

print("===== PLOTTING COMPLETED =====")
