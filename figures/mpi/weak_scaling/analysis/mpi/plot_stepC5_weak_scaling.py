import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

BASE = Path("data/mpi/weak_scaling")
OUT = Path("results/analysis/mpi/weak_scaling_figures")
OUT.mkdir(parents=True, exist_ok=True)

RANKS = [1, 2, 4, 8, 16, 32, 64, 128, 256]

STYLES = {
    "baseline": {
        "color": "#205781",
        "marker": "o",
        "linestyle": "-",
        "label": "Baseline",
    },
    "overlap": {
        "color": "#D87535",
        "marker": "s",
        "linestyle": "--",
        "label": "Overlap",
    },
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})

def load_csv(filename):
    with (BASE / filename).open(newline="") as f:
        return list(csv.DictReader(f))

summary = load_csv("weak_scaling_summary.csv")
iteration = load_csv("weak_scaling_iteration_analysis.csv")
communication = load_csv("weak_scaling_communication_summary.csv")
comparison = load_csv("weak_scaling_overlap_comparison.csv")

expected_counts = [
    (summary, 36),
    (iteration, 36),
    (communication, 36),
    (comparison, 18),
]

for rows, expected in expected_counts:
    if len(rows) != expected:
        raise SystemExit(
            f"Unexpected row count: {len(rows)}, expected {expected}"
        )

def get_series(rows, dim, comm, field):
    subset = [
        r for r in rows
        if int(r["dim"]) == dim and r["comm"] == comm
    ]

    ranks = [int(r["ranks"]) for r in subset]

    if len(subset) != len(RANKS) or sorted(ranks) != RANKS:
        raise SystemExit(
            f"Incomplete or duplicate ranks: dim={dim}, comm={comm}"
        )

    lookup = {
        int(r["ranks"]): float(r[field])
        for r in subset
    }

    return [lookup[p] for p in RANKS]

def save(fig, name):
    fig.tight_layout()

    for ext in ("png",):
        path = OUT / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print("CREATED:", path)

    plt.close(fig)

def configure_x(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(RANKS)
    ax.set_xticklabels([str(p) for p in RANKS])
    ax.set_xlabel("MPI Ranks")
    ax.grid(axis="y", linestyle=":", alpha=0.45)

def plot_lines(ax, rows, dim, field, scale=1.0):
    for comm, style in STYLES.items():
        values = [
            value * scale
            for value in get_series(rows, dim, comm, field)
        ]

        ax.plot(
            RANKS,
            values,
            linewidth=2,
            markersize=5,
            **style,
        )

def single_efficiency(dim):
    fig, ax = plt.subplots(figsize=(7.4, 4.9))

    plot_lines(
        ax,
        summary,
        dim,
        "weak_efficiency_percent",
    )

    configure_x(ax)
    ax.axhline(
        100,
        color="gray",
        linestyle=":",
        linewidth=1.2,
        label="Ideal",
    )

    ax.set_ylim(bottom=0)
    ax.set_ylabel("Weak Scaling Efficiency (%)")
    ax.set_title(f"{dim}D Weak Scaling Efficiency")
    ax.legend()

    save(fig, f"figure_{dim + 1:02d}_{dim}d_efficiency")

def dual_panel(number, name, rows, field, ylabel,
               log_y=False, scale=1.0):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.6, 4.8),
        sharey=False,
    )

    for ax, dim in zip(axes, (2, 3)):
        plot_lines(ax, rows, dim, field, scale)

        configure_x(ax)

        if log_y:
            ax.set_yscale("log")

        ax.set_title(f"{dim}D")
        ax.set_ylabel(ylabel)
        ax.legend()

    save(fig, f"figure_{number:02d}_{name}")

def overlap_improvement():
    fig, axes = plt.subplots(
        1, 2,
        figsize=(11.6, 4.8),
    )

    for ax, dim in zip(axes, (2, 3)):
        subset = [
            r for r in comparison
            if int(r["dim"]) == dim
        ]

        lookup = {
            int(r["ranks"]): float(
                r["runtime_improvement_percent"]
            )
            for r in subset
        }

        if len(subset) != len(RANKS) or set(lookup) != set(RANKS):
            raise SystemExit(
                f"Incomplete comparison data for {dim}D"
            )

        values = [lookup[p] for p in RANKS]

        colors = [
            "#238B65" if value >= 0 else "#C85A54"
            for value in values
        ]

        ax.bar(
            range(len(RANKS)),
            values,
            color=colors,
            width=0.68,
        )

        ax.axhline(0, color="black", linewidth=0.9)
        ax.set_xticks(range(len(RANKS)))
        ax.set_xticklabels([str(p) for p in RANKS])
        ax.set_xlabel("MPI Ranks")
        ax.set_ylabel("Runtime Improvement (%)")
        ax.set_title(f"{dim}D")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.set_axisbelow(True)

    save(fig, "figure_06_overlap_improvement")

# Figure 03 and 04
single_efficiency(2)
single_efficiency(3)

# Figure 05
dual_panel(
    5,
    "runtime",
    summary,
    "median_runtime_s",
    "Median Runtime (s)",
    log_y=True,
)

# Figure 06
overlap_improvement()

# Figure 07
dual_panel(
    7,
    "iterations",
    summary,
    "iterations",
    "CG Iterations",
    log_y=True,
)

# Figure 08
dual_panel(
    8,
    "per_iteration",
    iteration,
    "time_per_iteration_s",
    "Time per Iteration (ms)",
    log_y=True,
    scale=1000,
)

# Figure 09
dual_panel(
    9,
    "halo",
    communication,
    "median_halo_s",
    "Median Halo Time (s)",
    log_y=True,
)

# Figure 10
dual_panel(
    10,
    "allreduce",
    communication,
    "median_allreduce_s",
    "Median Allreduce Time (s)",
    log_y=True,
)

print()
print("===== FINAL STATUS =====")
print("EIGHT WEAK SCALING FIGURES GENERATED")
