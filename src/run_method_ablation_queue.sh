#!/usr/bin/env bash
set -euo pipefail

export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

pids=()
for seed in 0 1 2; do
  python3 artifact/src/run_fs.py \
    --out results --seed "${seed}" --tag objabl \
    --methods semantic_aug,semantic_align \
    > "method_ablation_s${seed}.log" 2>&1 &
  pids+=("$!")
done

for pid in "${pids[@]}"; do
  wait "${pid}"
done

echo "method objective ablations complete"
