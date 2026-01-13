#!/usr/bin/env bash
set -euo pipefail

COORD_ROOT=${1:-"outputs/manifests"}
OUT_DIR=${2:-"outputs/recognition"}

python -m src.recognition.train_from_coords \
  --coord_root "$COORD_ROOT" \
  --out_dir "$OUT_DIR" \
  --epochs 10 \
  --batch 64 \
  --lr 1e-3 \
  --img_size 64 \
  --num_workers 2 \
  --seed 42
