"""Build train/val/test manifests from coordinate .npy files."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build coordinate manifests")
    parser.add_argument(
        "--coord_root",
        type=str,
        required=True,
        help="Root directory containing train/val/test subfolders",
    )
    parser.add_argument("--out_dir", type=str, required=True, help="Output directory for manifests")
    parser.add_argument("--num_classes", type=int, default=None, help="Optional max number of classes")
    parser.add_argument("--max_per_class", type=int, default=None, help="Optional max samples per class per split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    return parser.parse_args()


def _collect_classes(train_root: Path, num_classes: int | None) -> List[str]:
    classes = sorted([p.name for p in train_root.iterdir() if p.is_dir()])
    if num_classes is not None:
        classes = classes[:num_classes]
    return classes


def main() -> None:
    args = parse_args()
    coord_root = Path(args.coord_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not coord_root.exists():
        raise FileNotFoundError(f"coord_root not found: {coord_root}")

    split_roots = {
        "train": coord_root / "train",
        "val": coord_root / "val",
        "test": coord_root / "test",
    }
    for split, split_root in split_roots.items():
        if not split_root.is_dir():
            raise FileNotFoundError(f"Missing {split} directory: {split_root}")

    classes = _collect_classes(split_roots["train"], args.num_classes)
    if not classes:
        raise RuntimeError(f"No class folders found under {split_roots['train']}")

    rng = random.Random(args.seed)
    rows: Dict[str, List[Dict[str, str | int]]] = {"train": [], "val": [], "test": []}

    for label, class_name in enumerate(classes):
        for split, split_root in split_roots.items():
            class_dir = split_root / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Missing class folder for {split}: {class_dir}")
            files = sorted(class_dir.glob("*.npy"))
            rng.shuffle(files)
            if args.max_per_class is not None:
                files = files[: args.max_per_class]
            for path in files:
                rows[split].append(
                    {
                        "path": str(path.resolve()),
                        "label": label,
                        "class": class_name,
                    }
                )

    if not rows["train"]:
        raise RuntimeError(
            "Train manifest would be empty. "
            f"Check that .npy files exist under {split_roots['train']}"
        )

    for split, split_rows in rows.items():
        rng.shuffle(split_rows)
        out_path = out_dir / f"manifest_{split}.csv"
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["path", "label", "class"])
            writer.writeheader()
            writer.writerows(split_rows)

    (out_dir / "classes.txt").write_text("\n".join(classes))
    print(f"Wrote manifests to {out_dir}")


if __name__ == "__main__":
    main()
