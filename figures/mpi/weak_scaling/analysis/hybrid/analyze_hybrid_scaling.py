import csv
import math
from pathlib import Path

INDIR = Path("results/analysis/hybrid")
INPUT = INDIR / "hybrid_formal_summary.csv"

SCALING = INDIR / "hybrid_strong_scaling.csv"
SUMMARY = INDIR / "hybrid_scaling_summary.csv"

if SCALING.exists() or SUMMARY.exists():
    raise SystemExit("Output already exists; refusing to overwrite")

with INPUT.open(newline="") as f:
    rows = list(csv.DictReader(f))

assert len(rows) == 48

lookup = {}

for r in rows:
    cores = int(r["cores"])
    dim = int(r["dim"])
    threads = int(r["threads"])
    comm = r["comm"]

    key = (dim, comm, threads, cores)

    if key in lookup:
        raise SystemExit(f"Duplicate configuration: {key}")

    runtime = float(r["median_runtime_s"])

    assert math.isfinite(runtime) and runtime > 0
    assert int(r["ranks"]) * threads == cores

    lookup[key] = runtime

scaling_rows = []
summary_rows = []

print("===== STEP C6.2.6.3 HYBRID STRONG SCALING =====")

for dim in (2, 3):
    for comm in ("baseline", "overlap"):
        for threads in (1, 2, 4):

            reference_key = (dim, comm, threads, 4)
            t4 = lookup[reference_key]

            print(
                f"\nDIM={dim} COMM={comm} THREADS={threads}"
            )
            print(
                f"{'CORES':>6} {'RANKS':>6} "
                f"{'RUNTIME(s)':>12} "
                f"{'SPEEDUP':>10} {'EFF%':>10}"
            )

            for cores in (4, 8, 16, 32):
                key = (dim, comm, threads, cores)
                runtime = lookup[key]

                ranks = cores // threads
                speedup = t4 / runtime
                efficiency = 100 * speedup / (cores / 4)

                result = {
                    "dim": dim,
                    "grid": 512 if dim == 2 else 64,
                    "comm": comm,
                    "threads": threads,
                    "cores": cores,
                    "ranks": ranks,
                    "median_runtime_s": runtime,
                    "reference_cores": 4,
                    "reference_runtime_s": t4,
                    "relative_speedup": speedup,
                    "relative_efficiency_percent": efficiency,
                }

                scaling_rows.append(result)

                print(
                    f"{cores:>6} {ranks:>6} "
                    f"{runtime:>12.6f} "
                    f"{speedup:>10.3f} "
                    f"{efficiency:>10.2f}"
                )

            t32 = lookup[(dim, comm, threads, 32)]
            s32 = t4 / t32
            e32 = 100 * s32 / 8

            summary_rows.append({
                "dim": dim,
                "comm": comm,
                "threads": threads,
                "reference_cores": 4,
                "target_cores": 32,
                "runtime_4_s": t4,
                "runtime_32_s": t32,
                "speedup_4_to_32": s32,
                "efficiency_4_to_32_percent": e32,
            })

assert len(scaling_rows) == 48
assert len(summary_rows) == 12

def write_csv(path, data):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(data[0].keys())
        )
        writer.writeheader()
        writer.writerows(data)

write_csv(SCALING, scaling_rows)
write_csv(SUMMARY, summary_rows)

print("\n===== 4-TO-32 CORE SCALING SUMMARY =====")

print(
    f"{'DIM':>4} {'COMM':<9} {'THREADS':>7} "
    f"{'SPEEDUP':>10} {'EFF%':>10}"
)

for r in summary_rows:
    print(
        f"{r['dim']:>4} "
        f"{r['comm']:<9} "
        f"{r['threads']:>7} "
        f"{r['speedup_4_to_32']:>10.3f} "
        f"{r['efficiency_4_to_32_percent']:>10.2f}"
    )

print("\n===== OUTPUT FILES =====")
print(SCALING)
print(SUMMARY)

print("\n===== FINAL STATUS =====")
print("HYBRID STRONG SCALING ANALYSIS COMPLETED")
