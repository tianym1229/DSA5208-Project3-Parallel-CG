import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

RAW = Path("data/serial/results_1518159.csv")
SUMMARY = Path("data/serial/serial_summary.csv")

with RAW.open(newline="") as f:
    rows = list(csv.DictReader(f))

with SUMMARY.open(newline="") as f:
    summaries = list(csv.DictReader(f))

assert len(rows) == 50
assert len(summaries) == 10

groups = defaultdict(list)

for r in rows:
    dim = int(r["dim"])
    grid = int(r["grid"])
    repeat = int(r["repeat"])
    n = int(r["n"])
    iterations = int(r["iterations"])
    runtime = float(r["runtime_s"])
    residual = float(r["relative_residual"])
    error = float(r["relative_error"])

    assert dim in (2, 3)
    assert grid >= 2
    assert n == grid ** dim
    assert 1 <= repeat <= 5
    assert iterations > 0
    assert math.isfinite(runtime) and runtime > 0
    assert math.isfinite(residual)
    assert math.isfinite(error)
    assert residual < 1e-8
    assert int(r["converged"]) == 1

    groups[(dim, grid)].append(r)

assert len(groups) == 10

summary_map = {
    (int(r["dim"]), int(r["grid"])): r
    for r in summaries
}

assert set(groups) == set(summary_map)

for key, data in sorted(groups.items()):
    repeats = sorted(int(r["repeat"]) for r in data)
    assert repeats == [1, 2, 3, 4, 5]

    iterations = {
        int(r["iterations"]) for r in data
    }
    assert len(iterations) == 1

    values = [float(r["runtime_s"]) for r in data]

    median = statistics.median(values)
    mean = statistics.mean(values)
    std = statistics.stdev(values)
    cv = 100 * std / mean

    s = summary_map[key]

    checks = {
        "median_s": median,
        "mean_s": mean,
        "std_s": std,
        "cv_percent": cv,
        "min_s": min(values),
        "max_s": max(values),
    }

    for field, calculated in checks.items():
        recorded = float(s[field])
        assert math.isclose(
            calculated,
            recorded,
            rel_tol=1e-9,
            abs_tol=1e-12
        ), f"{key}: mismatch in {field}"

    assert int(s["repeats"]) == 5
    assert int(s["iterations"]) == next(iter(iterations))
    assert int(s["n"]) == int(data[0]["n"])

    expected_high = int(cv > 20)
    assert int(s["high_variability"]) == expected_high

    print(
        f"PASS dim={key[0]} grid={key[1]} "
        f"median={median:.6f}s CV={cv:.2f}%"
    )

print("\n===== SERIAL AUDIT SUMMARY =====")
print("Raw runs:", len(rows))
print("Configurations:", len(groups))
print("All runs converged: YES")
print("Repeat coverage: PASS")
print("Statistical consistency: PASS")
print("STEP B8.3 SERIAL DATA AUDIT COMPLETED")
