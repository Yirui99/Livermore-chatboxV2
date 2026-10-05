#!/usr/bin/env bash
# Run every benchmark sequentially (no overlap, so they don't contend for the
# GPU/CPU) and save raw stdout to benchmarks/<outdir>/<name>.txt.
#   bash benchmarks/run_all.sh baseline
set -u
OUT="benchmarks/${1:?usage: run_all.sh <outdir>}"
PY="${PYTHON:-.venv/bin/python}"
mkdir -p "$OUT"
export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
{
  echo "date:    $(date -u +%FT%TZ)"
  echo "commit:  $(git rev-parse --short HEAD)$(git diff --quiet || echo -dirty)"
  echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null) / $(sysctl -n hw.memsize | awk '{print $1/2^30 " GB"}')"
  echo "python:  $($PY -c 'import sys;print(sys.version.split()[0])')  ($PY)"
  $PY -c 'import torch,faiss,sentence_transformers as s,mlx_lm;print("libs:    torch",torch.__version__,"faiss",faiss.__version__,"st",s.__version__,"mlx_lm",mlx_lm.__version__)'
} > "$OUT/env.txt"

run() {  # name, command...
  local name=$1; shift
  echo ">>> $name"
  local t0=$(date +%s)
  "$@" > "$OUT/$name.txt" 2>&1
  echo "    exit=$? ($(( $(date +%s) - t0 ))s)"
}
B=${BENCH_DIR:-benchmarks}
T=${OPS_DIR:-benchmarks}
run rag_latency      $PY $B/bench_rag_latency.py
run ops_correctness  $PY $T/test_ops_correctness.py
run matmul           $PY $T/bench_matmul.py
run mlx_4bit         $PY $B/benchmark_mlx_4bit.py
run mps_median       $PY $B/benchmark_mps_median.py
run rag_generation   $PY $B/benchmark_rag.py
run strategy         $PY ${STRATEGY_DIR:-examples/backtest}/test_strategy.py
echo done
