from pathlib import Path
import csv
import sys

COORD_ROOT = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
OUT_DIR.mkdir(parents=True, exist_ok=True)

for split in ["train", "val", "test"]:
    split_dir = COORD_ROOT / split
    if not split_dir.exists():
        print(f"[WARN] missing split folder: {split_dir}")
        continue

    # class names = folder names under split
    classes = sorted([p.name for p in split_dir.iterdir() if p.is_dir()])
    class_to_idx = {c:i for i,c in enumerate(classes)}

    out_csv = OUT_DIR / f"manifest_{split}.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "class_name"])
        n = 0
        for c in classes:
            cdir = split_dir / c
            for fp in cdir.rglob("*.npy"):
                w.writerow([str(fp), class_to_idx[c], c])
                n += 1
        print(f"{split}: {n} files, {len(classes)} classes -> {out_csv}")

    # Save class list for later
    (OUT_DIR / "classes.txt").write_text("\n".join(classes), encoding="utf-8")
