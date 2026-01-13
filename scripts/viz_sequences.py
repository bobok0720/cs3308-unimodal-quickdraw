"""Visualize coordinate sequences by rasterizing to PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from src.utils.rasterize_from_npy import rasterize_npy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rasterize a coordinate .npy to PNG")
    parser.add_argument("--npy", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--img_size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = rasterize_npy(args.npy, size=args.img_size)
    img_np = (img.squeeze(0).numpy() * 255).astype(np.uint8)
    Image.fromarray(img_np, mode="L").save(out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
