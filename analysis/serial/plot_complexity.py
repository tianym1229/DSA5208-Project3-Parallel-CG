import csv
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA = Path("data/serial/serial_summary.csv")
OUT = Path("results/analysis/serial/complexity_figures")
OUT.mkdir(parents=True, exist_ok=True)

with DATA.open(newline="") as f:
    rows = list(csv.DictReader(f))

for dim in (2, 3):
    data = sorted(
        (r for r in rows if int(r["dim"]) == dim),
        key=lambda r: int(r["n"])
    )

    assert len(data) == 5

    n = [int(r["n"]) for r in data]
    t = [float(r["median_s"]) for r in data]

    lx = [math.log(v) for v in n]
    ly = [math.log(v) for v in t]

    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)

    alpha = sum(
        (x-mx)*(y-my) for x,y in zip(lx,ly)
    ) / sum((x-mx)**2 for x in lx)

    intercept = my - alpha*mx

    predicted = [
        math.exp(intercept + alpha*math.log(v))
        for v in n
    ]

    sse = sum(
        (math.log(a)-math.log(b))**2
        for a,b in zip(t,predicted)
    )
    sst = sum((v-my)**2 for v in ly)
    r2 = 1-sse/sst

    fig, ax = plt.subplots(figsize=(7,5))

    ax.loglog(
        n, t, "o",
        markersize=7,
        label="Measured median runtime"
    )

    ax.loglog(
        n, predicted, "--",
        linewidth=2,
        label=f"Power-law fit: alpha={alpha:.3f}"
    )

    ax.set_xlabel("Number of unknowns (N)")
    ax.set_ylabel("Runtime (s)")
    ax.set_title(f"{dim}D Serial CG: Complexity Fit")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    ax.text(
        0.05, 0.95,
        f"alpha = {alpha:.4f}\nR² = {r2:.5f}",
        transform=ax.transAxes,
        va="top"
    )

    fig.tight_layout()

    fig.savefig(
        OUT / f"{dim}d_complexity_fit.png",
        dpi=200
    )
    fig.savefig(
        OUT / f"{dim}d_complexity_fit.pdf"
    )
    plt.close(fig)

    print(
        f"{dim}D: alpha={alpha:.6f}, R2={r2:.6f}"
    )

print("STEP B7.4 COMPLEXITY FIGURES COMPLETED")
