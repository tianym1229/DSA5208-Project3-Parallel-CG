import csv
import math
from pathlib import Path

INPUT = Path("data/serial/serial_summary.csv")

OUTPUT = Path("results/analysis/serial")
OUTPUT.mkdir(parents=True, exist_ok=True)

with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 10, "Expected 10 configurations"


def fit_power_law(x, y):
    """Fit log(y) = intercept + slope * log(x)."""
    assert len(x) == len(y) >= 3
    assert all(v > 0 for v in x)
    assert all(v > 0 for v in y)

    lx = [math.log(v) for v in x]
    ly = [math.log(v) for v in y]

    mx = sum(lx) / len(lx)
    my = sum(ly) / len(ly)

    sxx = sum((v - mx)**2 for v in lx)
    sxy = sum(
        (a - mx) * (b - my)
        for a, b in zip(lx, ly)
    )

    slope = sxy / sxx
    intercept = my - slope * mx

    predicted = [
        intercept + slope * v for v in lx
    ]

    sse = sum(
        (a - b)**2
        for a, b in zip(ly, predicted)
    )

    sst = sum(
        (v - my)**2 for v in ly
    )

    r2 = 1 - sse / sst if sst > 0 else 1.0

    return slope, intercept, r2


results = []

for dim in (2, 3):
    data = sorted(
        [r for r in rows if int(r["dim"]) == dim],
        key=lambda r: int(r["n"])
    )

    assert len(data) == 5

    n = [int(r["n"]) for r in data]
    runtime = [float(r["median_s"]) for r in data]
    iterations = [int(r["iterations"]) for r in data]

    alpha, intercept, r2 = fit_power_law(n, runtime)

    beta, _, iter_r2 = fit_power_law(
        n, iterations
    )

    work = [
        ni * ki
        for ni, ki in zip(n, iterations)
    ]

    gamma, _, work_r2 = fit_power_law(
        work, runtime
    )

    print(f"\n===== {dim}D SERIAL COMPLEXITY =====")
    print(f"N range: {min(n)} to {max(n)}")
    print(f"Runtime exponent alpha: {alpha:.6f}")
    print(f"Runtime fit R^2: {r2:.6f}")
    print(f"Iteration exponent beta: {beta:.6f}")
    print(f"Iteration fit R^2: {iter_r2:.6f}")
    print(f"Runtime vs N*k exponent: {gamma:.6f}")
    print(f"Work fit R^2: {work_r2:.6f}")

    print("\nGrid      N          Iterations   Median(s)")
    for r in data:
        print(
            f"{int(r['grid']):<9}"
            f"{int(r['n']):<11}"
            f"{int(r['iterations']):<13}"
            f"{float(r['median_s']):.6f}"
        )

    results.append({
        "dim": dim,
        "n_min": min(n),
        "n_max": max(n),
        "runtime_exponent": alpha,
        "runtime_r2": r2,
        "iteration_exponent": beta,
        "iteration_r2": iter_r2,
        "work_exponent": gamma,
        "work_r2": work_r2,
    })


out = OUTPUT / "complexity_summary.csv"

with out.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(results[0].keys())
    )
    writer.writeheader()
    writer.writerows(results)

print("\nSaved:", out)
print("STEP B7.3 COMPLEXITY ANALYSIS COMPLETED")
