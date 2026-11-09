import csv
import math
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path(
    "results/analysis/hybrid/"
    "hybrid_3d_scaling_diagnostics.csv"
)

OUTDIR = Path(
    "results/analysis/hybrid/figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

METRICS = [
    ("median_runtime_s", "Total runtime"),
    ("median_spmv_s", "SpMV"),
    ("median_halo_s", "Halo exchange"),
    ("median_allreduce_s", "Allreduce"),
    ("median_vector_s", "Vector operations"),
]

THREADS = [1, 2, 4]
COMMS = ["baseline", "overlap"]

COLORS = {
    1: "#2166AC",
    2: "#D6604D",
    4: "#4D9221",
}

LABELS = {
    1: "MPI-only (1 thread/rank)",
    2: "Hybrid (2 threads/rank)",
    4: "Hybrid (4 threads/rank)",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

print("===== STEP C6.2.7.8 H6 DATA VALIDATION =====")

with SOURCE.open(newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    columns = set(reader.fieldnames or [])

required = {
    "comm",
    "threads",
    "metric",
    "value_16_s",
    "value_32_s",
    "change_percent",
}

assert required <= columns, (
    f"Missing columns: {required - columns}"
)

assert len(rows) == 30, (
    f"Expected 30 records, got {len(rows)}"
)

lookup = {}

for row in rows:
    comm = row["comm"]
    threads = int(row["threads"])
    metric = row["metric"]

    assert comm in COMMS
    assert threads in THREADS
    assert metric in dict(METRICS)

    t16 = float(row["value_16_s"])
    t32 = float(row["value_32_s"])
    change = float(row["change_percent"])

    assert math.isfinite(t16) and t16 > 0
    assert math.isfinite(t32) and t32 > 0
    assert math.isfinite(change)

    calculated = 100.0 * (t32 - t16) / t16

    assert math.isclose(
        calculated,
        change,
        rel_tol=1e-9,
        abs_tol=1e-8,
    ), (
        f"Change mismatch: {comm}, "
        f"{threads}, {metric}"
    )

    key = (comm, threads, metric)

    assert key not in lookup, (
        f"Duplicate record: {key}"
    )

    lookup[key] = change

expected = {
    (comm, threads, metric)
    for comm in COMMS
    for threads in THREADS
    for metric, _ in METRICS
}

assert set(lookup) == expected

print("RECORDS:", len(rows))
print("CONFIGURATIONS:", len(lookup))
print("VALIDATION ERRORS: 0")

print("===== GENERATE H6 =====")

fig, axes = plt.subplots(
    1,
    2,
    figsize=(13.5, 5.5),
    sharey=True,
    constrained_layout=True,
)

positions = np.arange(len(METRICS))
bar_height = 0.23

for ax, comm in zip(axes, COMMS):

    for index, threads in enumerate(THREADS):

        values = [
            lookup[(comm, threads, metric)]
            for metric, _ in METRICS
        ]

        offsets = (
            positions
            + (index - 1) * bar_height
        )

        ax.barh(
            offsets,
            values,
            height=bar_height * 0.92,
            color=COLORS[threads],
            label=LABELS[threads],
        )

    ax.axvline(
        0,
        color="black",
        linewidth=1.3,
        linestyle=":",
    )

    ax.set_title(
        f"{comm.capitalize()} communication"
    )

    ax.set_xlabel(
        "Timing change from 16 to 32 cores (%)"
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [label for _, label in METRICS]
    )

    ax.invert_yaxis()

    ax.grid(
        axis="x",
        alpha=0.23,
        linewidth=0.8,
    )

    ax.set_axisbelow(True)

# Shared horizontal range makes the panels comparable.
all_changes = list(lookup.values())

lower = min(0, min(all_changes))
upper = max(0, max(all_changes))

margin = 0.08 * (upper - lower)

for ax in axes:
    ax.set_xlim(
        lower - margin,
        upper + margin,
    )

fig.suptitle(
    "3D Hybrid Scaling: Timing Changes "
    "from 16 to 32 Cores",
    fontsize=14,
    fontweight="bold",
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="outside lower center",
    ncol=3,
    frameon=False,
)

for extension in ("png", "pdf"):

    output = (
        OUTDIR /
        f"figure_H6_3d_timing_diagnostics.{extension}"
    )

    if output.exists():
        raise SystemExit(
            f"Refusing to overwrite: {output}"
        )

    fig.savefig(
        output,
        bbox_inches="tight",
    )

    print("CREATED:", output)

plt.close(fig)

print("===== H6 FIGURE GENERATION COMPLETED =====")
