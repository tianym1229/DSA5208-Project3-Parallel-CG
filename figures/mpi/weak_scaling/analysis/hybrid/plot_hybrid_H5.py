import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path(
    "results/analysis/hybrid/"
    "hybrid_formal_summary.csv"
)

OUTDIR = Path("results/analysis/hybrid/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

CORES = [4, 8, 16, 32]
THREADS = [1, 2, 4]

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

print("===== STEP C6.2.7.7 H5 DATA VALIDATION =====")

with SOURCE.open(newline="") as f:
    reader = csv.DictReader(f)
    print("CSV COLUMNS:", reader.fieldnames)
    rows = list(reader)

assert len(rows) == 48, (
    f"Expected 48 configurations, got {len(rows)}"
)

required = {
    "cores", "dim", "comm", "ranks",
    "threads", "median_runtime_s"
}

missing = required - set(reader.fieldnames)
assert not missing, f"Missing columns: {missing}"

lookup = {}

for row in rows:
    cores = int(row["cores"])
    dim = int(row["dim"])
    ranks = int(row["ranks"])
    threads = int(row["threads"])
    comm = row["comm"]
    runtime = float(row["median_runtime_s"])

    assert cores in CORES
    assert dim in (2, 3)
    assert threads in THREADS
    assert comm in ("baseline", "overlap")
    assert ranks * threads == cores
    assert math.isfinite(runtime) and runtime > 0

    key = (cores, dim, threads, comm)

    assert key not in lookup, f"Duplicate: {key}"
    lookup[key] = runtime

assert len(lookup) == 48

comparison = {}

for dim in (2, 3):
    for threads in THREADS:
        for cores in CORES:

            baseline = lookup[
                (cores, dim, threads, "baseline")
            ]

            overlap = lookup[
                (cores, dim, threads, "overlap")
            ]

            improvement = (
                100.0 * (baseline - overlap) / baseline
            )

            comparison[(dim, threads, cores)] = improvement

assert len(comparison) == 24

print("CONFIGURATIONS:", len(rows))
print("BASELINE/OVERLAP PAIRS:", len(comparison))

print("===== H5 PERFORMANCE SUMMARY =====")

for dim in (2, 3):
    for threads in THREADS:
        for cores in CORES:
            value = comparison[(dim, threads, cores)]
            print(
                f"DIM={dim} CORES={cores:2d} "
                f"THREADS={threads} "
                f"OVERLAP_IMPROVEMENT={value:+.2f}%"
            )

print("===== GENERATE H5 FIGURE =====")

fig, axes = plt.subplots(
    1, 2,
    figsize=(12.5, 5.0),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax, dim in zip(axes, (2, 3)):

    grid = 512 if dim == 2 else 64

    for threads in THREADS:

        values = [
            comparison[(dim, threads, cores)]
            for cores in CORES
        ]

        ax.plot(
            CORES,
            values,
            color=COLORS[threads],
            linestyle="-",
            marker="o",
            linewidth=1.9,
            markersize=5.5,
            label=LABELS[threads],
        )

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="Baseline reference (0%)",
    )

    ax.set_title(
        f"{dim}D Grid {grid} "
        f"(N = {grid ** dim:,})"
    )

    ax.set_xlabel("Total CPU cores")
    ax.set_ylabel("Overlap runtime improvement (%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(CORES)
    ax.set_xticklabels([str(c) for c in CORES])
    ax.grid(True, alpha=0.23, linewidth=0.8)

fig.suptitle(
    "Baseline vs Overlap: Runtime Improvement",
    fontsize=14,
    fontweight="bold",
)

handles, labels = axes[0].get_legend_handles_labels()

fig.legend(
    handles,
    labels,
    loc="outside lower center",
    ncol=2,
    frameon=False,
)

for extension in ("png", "pdf"):

    output = (
        OUTDIR /
        f"figure_H5_baseline_vs_overlap.{extension}"
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

print("===== H5 FIGURE GENERATION COMPLETED =====")
