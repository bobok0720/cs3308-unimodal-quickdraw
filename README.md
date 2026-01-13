# CS3308 QuickDraw: Recognition + Generation

This repo provides an end-to-end pipeline for coordinate-based **recognition** and **conditional generation** using the QuickDraw414k coordinate dataset. It includes scripts, notebooks, and utilities that run both **locally** and in **Google Colab** without hard-coded repo paths.

## Quickstart (Colab)

1. **Clone this repo** into your Drive or local Colab session.
2. **Open** `notebooks/99_full_pipeline_clean.ipynb`.
3. **Run Section A (Setup)** and point `COORD_ROOT` at your data.
4. **Run Sections B–E** end-to-end.

**Expected outputs**:
- Manifests under `outputs/manifests/`
- Recognition checkpoint under `outputs/recognition/best_recognizer.pt`
- Generation checkpoint under `outputs/generation/best_condlstm.pt`
- Generated samples under `figures/gen_samples/<class>/*.png`
- Evaluation metrics under `outputs/eval/metrics.json` and `outputs/eval/confusion_matrix.png`

## Colab Smoke Test (Recognition)

```bash
# Clone repo
repo_root="$HOME/cs3308-unimodal-quickdraw"
git clone https://github.com/CS3308/cs3308-unimodal-quickdraw.git "$repo_root"
cd "$repo_root"

# Install deps
python -m pip install -r requirements.txt

# Mount Drive (for data access)
python - <<'PY'
from google.colab import drive

drive.mount("/content/drive")
PY

# Point this to your QuickDraw414k coordinate_files folder
COORD_ROOT=/path/to/QuickDraw414k/coordinate_files

# Build manifests (small subset)
python scripts/build_coord_manifest.py \
  --coord_root "$COORD_ROOT" \
  --out_dir outputs/manifests \
  --num_classes 5 \
  --max_per_class 300

# Train recognizer (2 epochs)
python -m src.recognition.train_from_coords \
  --rec_out outputs/manifests \
  --out_dir outputs/recognition \
  --epochs 2
```

## Local Run

```bash
python -m pip install -r requirements.txt

# Build manifests
python scripts/build_coord_manifest.py \
  --coord_root /path/to/coordinate_files \
  --out_dir outputs/manifests

# Train recognizer
python -m src.recognition.train_from_coords \
  --rec_out outputs/manifests \
  --out_dir outputs/recognition

# Train generator + sample
python -m src.generation.train_lstm \
  --rec_out outputs/manifests \
  --out_dir outputs/generation
python -m src.generation.sample_condlstm \
  --ckpt outputs/generation/best_condlstm.pt \
  --out_dir figures

# Evaluate generated samples with recognizer
python -m src.generation.eval_with_recognizer \
  --rec_ckpt outputs/recognition/best_recognizer.pt \
  --gen_root figures/gen_samples \
  --out_dir outputs/eval
```

## Outputs

```
outputs/
  manifests/
    manifest_train.csv
    manifest_val.csv
    manifest_test.csv
    classes.txt
  recognition/
    best_recognizer.pt
    history.json
  generation/
    best_condlstm.pt
    history.json
  eval/
    metrics.json
    confusion_matrix.png
figures/
  gen_samples/<class>/<class>_<k>.png
```

## Troubleshooting

- **Drive mountpoint not empty**: In Colab, re-mount with `force_remount=True` or remove old mount folders.
- **Unicode decode errors in np.load**: The loader uses `allow_pickle=True` with latin1 fallback (see `src/utils/rasterize_from_npy.py`).
- **Object arrays / weird shapes**: The loader tries to unwrap object arrays and `(1, T, 4)` wrappers.
- **Import errors in notebooks**: Run from repo root or use the setup cell to `cd` to the repo root before importing `src.*`.

## Repository Structure

```
cs3308-unimodal-quickdraw/
  README.md
  requirements.txt
  .gitignore
  src/
    recognition/
    generation/
    utils/
  scripts/
  notebooks/
  configs/
  outputs/   (gitignored)
  figures/   (gitignored)
```

## Notes

- Models are designed to run on CPU or GPU automatically.
- Use `python -m ...` to avoid hard-coded paths.
