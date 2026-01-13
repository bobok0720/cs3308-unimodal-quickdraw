#!/usr/bin/env bash
set -euo pipefail

REC_CKPT=${1:-"outputs/recognition/best_recognizer.pt"}
GEN_ROOT=${2:-"figures/gen_samples"}
OUT_DIR=${3:-"outputs/eval"}

python -m src.generation.eval_with_recognizer \
  --rec_ckpt "$REC_CKPT" \
  --gen_root "$GEN_ROOT" \
  --out_dir "$OUT_DIR"
