#!/usr/bin/env bash
set -euo pipefail

REC_OUT=${1:-"outputs/manifests"}
OUT_DIR=${2:-"outputs/generation"}

python -m src.generation.train_lstm \
  --rec_out "$REC_OUT" \
  --out_dir "$OUT_DIR" \
  --epochs 10 \
  --batch 64 \
  --lr 1e-3 \
  --seed 42

python -m src.generation.sample_condlstm \
  --ckpt "$OUT_DIR/best_condlstm.pt" \
  --out_dir "figures" \
  --n_per_class 5 \
  --T 100 \
  --temp 0.7 \
  --img_size 64
