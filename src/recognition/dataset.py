"""Datasets for coordinate-based recognition."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.rasterize_from_npy import rasterize_npy


@dataclass
class ManifestSample:
    path: str
    label: int
    class_name: str
    split: str


def load_manifest(csv_path: Path) -> List[ManifestSample]:
    df = pd.read_csv(csv_path)
    required = {"path", "label", "class_name", "split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    return [
        ManifestSample(row.path, int(row.label), row.class_name, row.split)
        for row in df.itertuples(index=False)
    ]


class CoordRasterDataset(Dataset):
    """Dataset that rasterizes coordinate .npy files on the fly."""

    def __init__(self, samples: List[ManifestSample], img_size: int = 64):
        self.samples = samples
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        image = rasterize_npy(sample.path, size=self.img_size)
        return image, sample.label
