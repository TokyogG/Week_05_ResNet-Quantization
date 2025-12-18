#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-mobilenet_v2}"
ITERS="${2:-200}"
WARMUP="${3:-50}"
CPU_CORE="${CPU_CORE:-2}"

PY="${PYTHON:-$(command -v python3)}"

echo "[run_bench] PY=$PY"
echo "[run_bench] MODEL=$MODEL ITERS=$ITERS WARMUP=$WARMUP CPU_CORE=$CPU_CORE"

bash scripts/sanity_env.sh

taskset -c "${CPU_CORE}" \
  nice -n -10 \
  "$PY" bench_pi5_inference.py \
    --model "${MODEL}" \
    --iters "${ITERS}" \
    --warmup "${WARMUP}" \
    --threads 1 \
    --interop_threads 1 \
    --pin_core "${CPU_CORE}" \
    --engine qnnpack \
    --out_dir results \
    --tag locked
