#!/usr/bin/env bash
set -euo pipefail

for method in gate_robust gate_srr gate_full; do
  for seed in 0 1 2; do
    python3 artifact/src/new_experiments.py pgd-table \
      --n 1000 --methods "${method}" --seeds "${seed}"
  done
done

python3 artifact/src/new_experiments.py modern-pgd \
  --n 1000 --methods semantic_tc,gate_full --seeds 0 \
  --generators Midjourney,stable_diffusion_v_1_5 --eps 4 --steps 40
