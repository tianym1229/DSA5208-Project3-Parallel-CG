#!/usr/bin/env python3

import csv
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

SOURCE = ROOT / (
    "data/gpu/gpu_benchmark_summary.csv"
)

OUT = ROOT / "results/analysis/gpu/figures"

with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))

rows.sort(key=lambda r: int(r["dim"]))

assert len(rows) == 2
assert [(int(r["dim"]), int(r["grid"])) for r in rows] == [
    (2, 512), (3, 64)
]

labels = [
    f"{r['dim']}D Grid {r['grid']}"
    for r in rows
]

def values(field):
    return np.array([
        1000 * float(r[field])
        for r in rows
    ])

fig, axes = plt.subplots(
    2, 1,
    figsize=(9, 8),
    constrained_layout=True
)

width = 0.34

groups = [
    (
        axes[0],
        [
            ("Solver", "median_solver_runtime_s"),
            ("End-to-End", "median_end_to_end_s"),
        ],
        "A. Overall Runtime"
    ),
    (
        axes[1],
        [
            ("H2D", "median_h2d_s"),
            ("D2H", "median_d2h_s"),
            ("Validation", "median_validation_s"),
        ],
        "B. Transfer and Validation"
    ),
]

for ax, metrics, title in groups:
    x = np.arange(len(metrics))

    for i, label in enumerate(labels):
        data = np.array([
            1000 * float(rows[i][field])
            for _, field in metrics
        ])

        bars = ax.bar(
            x + (i - 0.5) * width,
            data,
            width,
            label=label
        )

        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                (
                    bar.get_x() + bar.get_width() / 2,
                    height
                ),
                ha="center",
                va="bottom",
                fontsize=9,
                xytext=(0, 3),
                textcoords="offset points"
            )

    ax.set_xticks(
        x,
        [name for name, _ in metrics]
    )
    ax.set_ylabel("Median runtime (ms)")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_yscale("log")

    lower = 10 if ax is axes[0] else 0.5
    upper = 1000 if ax is axes[0] else 5

    ax.set_ylim(lower, upper)
    ax.set_ylabel("Median runtime (ms, log scale)")

axes[0].legend()

fig.suptitle(
    "GPU Runtime Metrics — NVIDIA A40",
    fontsize=15
)

for ext in ("png", "pdf"):
    path = OUT / f"figure_G3_gpu_runtime_components_v4_logbar.{ext}"

    if path.exists():
        raise FileExistsError(
            f"Refusing to overwrite: {path}"
        )

    fig.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )
    print("CREATED:", path)

plt.close(fig)

print("===== G3 V2 COMPLETED =====")
