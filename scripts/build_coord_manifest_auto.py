from pathlib import Path
import csv
import random
import sys

COORD_ROOT = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
VAL_FRAC = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05
SEED = 42

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Detect split folders
candidates = [p for p in COORD_ROOT.iterdir() if p.is_dir()]
split_names = sorted([p.name for p in candidates])
print("Detected split folders:", split_names)

# Helper: list classes under a split
def list_classes(split_dir: Path):
    return sorted([p.name for p in split_dir.iterdir() if p.is_dir()])

# Choose which folders to treat as train/val/test
train_name = "train" if (COORD_ROOT/"train").exists() else None
val_name   = "val"   if (COORD_ROOT/"val").exists() else ("valid" if (COORD_ROOT/"valid").exists() else None)
test_name  = "test"  if (COORD_ROOT/"test").exists() else None

if train_name is None:
    raise SystemExit("No train/ folder found. Please check coordinate_files structure.")

train_dir = COORD_ROOT / train_name
classes = list_classes(train_dir)
class_to_idx = {c:i for i,c in enumerate(classes)}
(OUT_DIR / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
print("Classes:", len(classes))

def write_manifest(split_dir: Path, split_label: str):
    out_csv = OUT_DIR / f"manifest_{split_label}.csv"
    n = 0
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class_name"])
        for c in classes:
            cdir = split_dir / c
            if not cdir.exists():
                continue
            for fp in cdir.rglob("*.npy"):
                w.writerow([str(fp), class_to_idx[c], c])
                n += 1
    print(f"{split_label}: {n} files -> {out_csv}")
    return out_csv, n

# Build train manifest
train_csv, train_n = write_manifest(train_dir, "train")

# Build test manifest if exists
if test_name is not None:
    test_dir = COORD_ROOT / test_name
    write_manifest(test_dir, "test")
else:
    print("[WARN] No test/ folder found. You will not be able to report test metrics yet.")

# Build val manifest:
# If an explicit val folder exists, use it; otherwise sample from train per class.
if val_name is not None:
    val_dir = COORD_ROOT / val_name
    write_manifest(val_dir, "val")
else:
    print(f"[INFO] No val/ folder found. Creating val split by sampling {VAL_FRAC*100:.1f}% from train per class.")

    rng = random.Random(SEED)
    train_rows = []
    with open(train_csv, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            train_rows.append(row)

    # Group by class_name
    by_class = {}
    for row in train_rows:
        by_class.setdefault(row["class_name"], []).append(row)

    val_rows = []
    new_train_rows = []
    for c, rows in by_class.items():
        rng.shuffle(rows)
        k = max(1, int(len(rows) * VAL_FRAC))
        val_rows.extend(rows[:k])
        new_train_rows.extend(rows[k:])

    # Write new manifests
    val_csv = OUT_DIR / "manifest_val.csv"
    with val_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class_name"])
        for row in val_rows:
            w.writerow([row["path"], row["label"], row["class_name"]])

    train_csv2 = OUT_DIR / "manifest_train.csv"
    with train_csv2.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class_name"])
        for row in new_train_rows:
            w.writerow([row["path"], row["label"], row["class_name"]])

    print(f"Rewrote train: {len(new_train_rows)} rows")
    print(f"Wrote val:     {len(val_rows)} rows -> {val_csv}")
