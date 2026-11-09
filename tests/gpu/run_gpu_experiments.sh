#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

OUT="$ROOT/results/analysis/gpu/execution"
mkdir -p "$OUT"

STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$OUT/gpu_experiment_${STAMP}.log"

exec > >(tee -a "$LOG") 2>&1

echo "===== GPU EXPERIMENTS ====="
date

echo "===== PROJECT ROOT ====="
echo "$ROOT"

echo "===== GPU NODE ====="
hostname

echo "===== GPU HARDWARE ====="
nvidia-smi

echo "===== CUDA ENVIRONMENT ====="
module load CUDA/12.4.0
nvcc --version

echo "===== GPU EXECUTABLE ====="
test -x build/cg_gpu
ls -lh build/cg_gpu

echo "===== CPU SOURCE INTEGRITY ====="
sha256sum -c data/serial/cpu_source_sha256.txt

echo "===== GPU CORRECTNESS TEST ====="
python3 -u tests/gpu/test_gpu_correctness.py

echo "===== GPU BENCHMARK ====="
python3 -u tests/gpu/benchmark_gpu.py

echo "===== EXPERIMENT COMPLETED ====="
date
