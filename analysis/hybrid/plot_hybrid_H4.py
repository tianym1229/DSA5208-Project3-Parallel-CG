import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SOURCE = Path(
    "results/analysis/hybrid/"
    "hybrid_configuration_comparison.csv"
)

OUTDIR = Path("results/analysis/hybrid/figures")
OUTDIR.mkdir(parents=True, exist_ok=True)

with SOURCE.open(newline="") as f:
    data = list(csv.DictReader(f))

assert len(data) == 32, "Expected 32 comparison records"

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
    2: "#D6604D",
    4: "#4D9221",
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

seen = set()

for row in data:
    cores = int(row["cores"])
    dim = int(row["dim"])
    threads = int(row["threads"])
    ranks = int(row["ranks"])
    comm = row["comm"]

    assert cores in CORES
    assert dim in (2, 3)
    assert threads in (2, 4)
    assert comm in ("baseline", "overlap")
    assert ranks * threads == cores

    key = (cores, dim, threads, comm)
    assert key not in seen
    seen.add(key)

    ref = float(row["pure_mpi_runtime_s"])
    hybrid = float(row["hybrid_runtime_s"])
    improvement = float(row["hybrid_improvement_percent"])

    assert math.isfinite(ref) and ref > 0
    assert math.isfinite(hybrid) and hybrid > 0
    assert math.isfinite(improvement)

    calculated = 100 * (ref - hybrid) / ref
    assert abs(calculated - improvement) < 1e-8

assert len(seen) == 32

fig, axes = plt.subplots(
    1, 2,
    figsize=(12.5, 5.0),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

for ax, dim in zip(axes, (2, 3)):
    grid = 512 if dim == 2 else 64

    for threads in (2, 4):
        for comm in ("baseline", "overlap"):
            rows = [
                r for r in data
                if int(r["dim"]) == dim
                and int(r["threads"]) == threads
                and r["comm"] == comm
            ]

            rows.sort(key=lambda r: int(r["cores"]))

            assert len(rows) == 4
            assert [int(r["cores"]) for r in rows] == CORES

            values = [
                float(r["hybrid_improvement_percent"])
                for r in rows
            ]

            ax.plot(
                CORES,
                values,
                color=COLORS[threads],
                linestyle=STYLES[comm],
                marker=MARKERS[comm],
                linewidth=1.9,
                markersize=5.5,
                label=f"{threads} threads/rank - {comm}",
            )

    ax.axhline(
        0,
        color="black",
        linestyle=":",
        linewidth=1.5,
        label="MPI-only reference (0%)",
    )

    ax.set_title(
        f"{dim}D Grid {grid} (N = {grid ** dim:,})"
    )
    ax.set_xlabel("Total CPU cores")
    ax.set_ylabel("Hybrid runtime improvement (%)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(CORES)
    ax.set_xticklabels([str(c) for c in CORES])
    ax.grid(True, alpha=0.23, linewidth=0.8)

fig.suptitle(
    "Hybrid vs MPI-only: Runtime Improvement",
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
    output = OUTDIR / f"figure_H4_hybrid_vs_mpi.{extension}"

    if output.exists():
        raise SystemExit(f"Refusing to overwrite: {output}")

    fig.savefig(output, bbox_inches="tight")
    print("CREATED:", output)

plt.close(fig)

print("===== H4 FIGURE GENERATION COMPLETED =====")
