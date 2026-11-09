import csv
import math
from pathlib import Path

BASE = Path("results/analysis/hybrid")
SOURCE = BASE / "hybrid_formal_summary.csv"
COMPARISON = BASE / "hybrid_configuration_comparison.csv"
DIAGNOSTICS = BASE / "hybrid_3d_scaling_diagnostics.csv"

if COMPARISON.exists() or DIAGNOSTICS.exists():
    raise SystemExit("Output exists; refusing to overwrite")

with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 48

lookup = {}
for row in rows:
    key = (
        int(row["cores"]),
        int(row["dim"]),
        row["comm"],
        int(row["threads"]),
    )
    assert key not in lookup
    lookup[key] = row

def metric(row, name):
    value = float(row[name])
    assert math.isfinite(value)
    return value

def runtime(row):
    return metric(row, "median_runtime_s")

comparison = []

print("===== STEP C6.2.6.4 HYBRID COMPARISON =====")
print("\n===== HYBRID VS PURE MPI =====")

for cores in (4, 8, 16, 32):
    for dim in (2, 3):
        for comm in ("baseline", "overlap"):
            reference = lookup[(cores, dim, comm, 1)]
            ref_time = runtime(reference)

            for threads in (2, 4):
                current = lookup[(cores, dim, comm, threads)]
                cur_time = runtime(current)

                improvement = 100 * (ref_time - cur_time) / ref_time

                comparison.append({
                    "cores": cores,
                    "dim": dim,
                    "comm": comm,
                    "threads": threads,
                    "ranks": cores // threads,
                    "pure_mpi_runtime_s": ref_time,
                    "hybrid_runtime_s": cur_time,
                    "hybrid_improvement_percent": improvement,
                })

                print(
                    f"CORES={cores:2d} DIM={dim} "
                    f"COMM={comm:<8} "
                    f"CONFIG={cores//threads}x{threads} "
                    f"IMPROVEMENT={improvement:+.2f}%"
                )

print("\n===== BASELINE VS OVERLAP =====")

for cores in (4, 8, 16, 32):
    for dim in (2, 3):
        for threads in (1, 2, 4):
            baseline = runtime(
                lookup[(cores, dim, "baseline", threads)]
            )
            overlap = runtime(
                lookup[(cores, dim, "overlap", threads)]
            )

            improvement = 100 * (baseline - overlap) / baseline

            print(
                f"CORES={cores:2d} DIM={dim} "
                f"CONFIG={cores//threads}x{threads} "
                f"OVERLAP_IMPROVEMENT={improvement:+.2f}%"
            )

diagnostics = []

metrics = [
    "median_runtime_s",
    "median_spmv_s",
    "median_halo_s",
    "median_allreduce_s",
    "median_vector_s",
]

print("\n===== 3D 16-TO-32 CORE DIAGNOSTICS =====")

for comm in ("baseline", "overlap"):
    for threads in (1, 2, 4):
        a = lookup[(16, 3, comm, threads)]
        b = lookup[(32, 3, comm, threads)]

        print(f"\nCOMM={comm} THREADS={threads}")

        for name in metrics:
            old = metric(a, name)
            new = metric(b, name)

            change = 100 * (new - old) / old if old > 0 else float("nan")

            diagnostics.append({
                "comm": comm,
                "threads": threads,
                "metric": name,
                "value_16_s": old,
                "value_32_s": new,
                "change_percent": change,
            })

            print(
                f"{name:<24} "
                f"16={old:.6f} "
                f"32={new:.6f} "
                f"CHANGE={change:+.2f}%"
            )

def save(path, data):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(data[0])
        )
        writer.writeheader()
        writer.writerows(data)

assert len(comparison) == 32
assert len(diagnostics) == 30

save(COMPARISON, comparison)
save(DIAGNOSTICS, diagnostics)

print("\n===== OUTPUT CHECK =====")
print("Hybrid comparisons:", len(comparison))
print("Timing diagnostics:", len(diagnostics))
print("Output:", COMPARISON)
print("Output:", DIAGNOSTICS)

print("\n===== FINAL STATUS =====")
print("HYBRID COMPARISON ANALYSIS COMPLETED")
