import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

source = Path("results/step5h_analysis/scaling_summary.csv")
outdir = Path("results/step5i_figures")
outdir.mkdir(parents=True, exist_ok=True)

with source.open(newline="") as f:
    rows = list(csv.DictReader(f))

styles = {
    "baseline": {"marker": "o", "linestyle": "-", "color": "#2563eb"},
    "overlap": {"marker": "s", "linestyle": "--", "color": "#ea580c"},
}

metrics = [
    ("median_s", "Runtime (s)", "runtime"),
    ("speedup", "Speedup", "speedup"),
    ("efficiency", "Parallel Efficiency (%)", "efficiency"),
]

for dim, grid in [(2, 512), (3, 64)]:
    for field, ylabel, suffix in metrics:

        fig, ax = plt.subplots(figsize=(7.5, 5))

        for comm in ["baseline", "overlap"]:
            data = sorted(
                [
                    r for r in rows
                    if int(r["dim"]) == dim
                    and r["comm"] == comm
                ],
                key=lambda r: int(r["ranks"])
            )

            x = [int(r["ranks"]) for r in data]
            y = [float(r[field]) for r in data]

            if field == "efficiency":
                y = [v * 100 for v in y]

            ax.plot(
                x, y,
                label=comm.capitalize(),
                linewidth=2,
                markersize=6,
                **styles[comm]
            )

        ranks = [1, 2, 4, 8, 16]

        if field == "speedup":
            ax.plot(
                ranks, ranks,
                linestyle=":",
                color="gray",
                label="Ideal Linear Speedup"
            )

        if field == "efficiency":
            ax.axhline(
                100,
                linestyle=":",
                color="gray",
                label="Ideal Efficiency"
            )

        ax.set_title(
            f"{dim}D Strong Scaling (Grid={grid})"
        )
        ax.set_xlabel("MPI Ranks")
        ax.set_ylabel(ylabel)
        ax.set_xticks(ranks)
        ax.grid(True, alpha=0.3)
        ax.legend()

        fig.tight_layout()

        name = f"{dim}d_{suffix}"

        fig.savefig(
            outdir / f"{name}.png",
            dpi=300,
            bbox_inches="tight"
        )

        fig.savefig(
            outdir / f"{name}.pdf",
            bbox_inches="tight"
        )

        plt.close(fig)
        print("Saved:", name)

print("STEP 5I FIGURE GENERATION COMPLETED")
