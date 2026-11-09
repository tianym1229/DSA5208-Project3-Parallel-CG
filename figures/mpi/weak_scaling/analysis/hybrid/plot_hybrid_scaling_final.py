import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

SOURCE = Path(
    "results/analysis/hybrid/"
    "hybrid_strong_scaling.csv"
)

OUTDIR = Path(
    "results/analysis/hybrid/figures"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

with SOURCE.open(newline="") as f:
    data = list(csv.DictReader(f))

assert len(data) == 48

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

STYLES = {
    "baseline": "-",
    "overlap": "--",
}

MARKERS = {
    "baseline": "o",
    "overlap": "s",
}

CORES = [4, 8, 16, 32]


def get_series(dim, comm, threads):
    rows = [
        r for r in data
        if int(r["dim"]) == dim
        and r["comm"] == comm
        and int(r["threads"]) == threads
    ]

    rows.sort(key=lambda r: int(r["cores"]))

    assert len(rows) == 4
    assert [int(r["cores"]) for r in rows] == CORES

    return rows


def draw_figure(filename, metric, ylabel, title,
                log_y=False, ideal=False):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12.5, 5.0),
        sharex=True,
        constrained_layout=True,
    )

    for ax, dim in zip(axes, (2, 3)):

        grid = 512 if dim == 2 else 64

        for threads in (1, 2, 4):
            for comm in ("baseline", "overlap"):

                rows = get_series(dim, comm, threads)

                x = [int(r["cores"]) for r in rows]
                y = [float(r[metric]) for r in rows]

                ax.plot(
                    x, y,
                    color=COLORS[threads],
                    linestyle=STYLES[comm],
                    marker=MARKERS[comm],
                    linewidth=1.9,
                    markersize=5.2,
                    label=f"{LABELS[threads]} - {comm}",
                )

        if ideal:
            if metric == "relative_speedup":
                ideal_y = [p / 4 for p in CORES]
            else:
                ideal_y = [100] * len(CORES)

            ax.plot(
                CORES,
                ideal_y,
                color="black",
                linestyle=":",
                linewidth=1.6,
                label="Ideal scaling",
            )

        ax.set_title(
            f"{dim}D Grid {grid} "
            f"(N = {grid ** dim:,})"
        )

        ax.set_xlabel("Total CPU cores")
        ax.set_ylabel(ylabel)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CORES)
        ax.set_xticklabels([str(c) for c in CORES])

        ax.grid(
            True,
            alpha=0.23,
            linewidth=0.8,
        )

        if log_y:
            ax.set_yscale("log")

            ticks = (
                [0.1, 0.2, 0.3, 0.5, 1.0]
                if dim == 2
                else [0.04, 0.06, 0.1, 0.2, 0.3]
            )

            ax.yaxis.set_major_locator(
                FixedLocator(ticks)
            )

            ax.yaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, pos: f"{value:g}"
                )
            )

            ax.minorticks_off()

        if metric == "relative_efficiency_percent":
            ax.axhline(
                100,
                color="black",
                linestyle=":",
                linewidth=1.6,
                zorder=0,
            )

    handles, labels = axes[0].get_legend_handles_labels()

    # Avoid duplicate ideal reference in efficiency legend.
    if metric == "relative_efficiency_percent":
        handles.append(
            plt.Line2D(
                [], [],
                color="black",
                linestyle=":",
                linewidth=1.6,
            )
        )
        labels.append("Ideal efficiency (100%)")

    fig.legend(
        handles,
        labels,
        loc="outside lower center",
        ncol=2,
        frameon=False,
    )

    fig.suptitle(
        title,
        fontsize=14,
        fontweight="bold",
    )

    for extension in ("png", "pdf"):
        output = OUTDIR / f"{filename}.{extension}"

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


print("===== FINAL HYBRID SCALING FIGURES =====")

draw_figure(
    "figure_H1_runtime",
    "median_runtime_s",
    "Median runtime (s)",
    "Hybrid Strong Scaling: Runtime",
    log_y=True,
)

draw_figure(
    "figure_H2_speedup",
    "relative_speedup",
    "Speedup relative to 4 cores",
    "Hybrid Strong Scaling: Relative Speedup",
    ideal=True,
)

draw_figure(
    "figure_H3_efficiency",
    "relative_efficiency_percent",
    "Relative efficiency (%)",
    "Hybrid Strong Scaling: Relative Efficiency",
)

print("===== FINAL FIGURES COMPLETED =====")
