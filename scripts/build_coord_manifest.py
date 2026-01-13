"""Build train/val/test manifests from coordinate .npy files."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build coordinate manifests")
    parser.add_argument("--coord_root", type=str, required=True, help="Root directory of coord class folders")
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for manifests")
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_classes", type=str, nargs="*", default=None, help="Optional class list for tiny split")
    parser.add_argument("--tiny_per_class", type=int, default=None, help="Max samples per class in tiny mode")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    coord_root = Path(args.coord_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not coord_root.exists():
        raise FileNotFoundError(f"coord_root not found: {coord_root}")

    classes = sorted([p.name for p in coord_root.iterdir() if p.is_dir()])
    if args.tiny_classes:
        classes = [c for c in classes if c in args.tiny_classes]
    if not classes:
        raise ValueError("No classes found")

    random.seed(args.seed)
    rows = {"train": [], "val": [], "test": []}

    for label, class_name in enumerate(classes):
        class_dir = coord_root / class_name
        files = sorted([p for p in class_dir.glob("*.npy")])
        if args.tiny_per_class:
            random.shuffle(files)
            files = files[: args.tiny_per_class]
        random.shuffle(files)
        n_total = len(files)
        n_val = int(n_total * args.val_frac)
        n_test = int(n_total * args.test_frac)
        n_train = n_total - n_val - n_test

        splits = (
            ["train"] * n_train
            + ["val"] * n_val
            + ["test"] * n_test
        )

        for path, split in zip(files, splits):
            rows[split].append(
                {
                    "path": str(path.resolve()),
                    "label": label,
                    "class_name": class_name,
                    "split": split,
                }
            )

    for split, split_rows in rows.items():
        out_path = out_dir / f"manifest_{split}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "class_name", "split"])
            writer.writeheader()
            writer.writerows(split_rows)

    (out_dir / "classes.txt").write_text("\n".join(classes))
    print(f"Wrote manifests to {out_dir}")


if __name__ == "__main__":
    main()
