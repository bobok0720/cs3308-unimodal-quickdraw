"""Dataset for conditional LSTM generation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils.rasterize_from_npy import extract_coords, safe_np_load


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


def coords_to_sketch5(coords: np.ndarray) -> np.ndarray:
    """Convert coords array to sketch-5 [dx,dy,p1,p2,p3]."""

    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        raise ValueError("coords must be (T,>=2)")

    xy = coords[:, :2].astype(np.float32)
    max_abs = float(np.max(np.abs(xy))) if xy.size else 0.0
    if max_abs > 20.0:
        deltas = np.diff(xy, axis=0, prepend=xy[:1])
    else:
        deltas = xy
        xy = np.cumsum(deltas, axis=0)

    pen_up = None
    if coords.shape[1] >= 3:
        pen_col = coords[:, 2]
        if np.isin(pen_col, [0, 1]).all():
            pen_up = pen_col.astype(np.int64)

    if pen_up is None and coords.shape[1] >= 4:
        pen_cols = coords[:, 2:4]
        if np.isin(pen_cols, [0, 1]).all():
            pen_down = pen_cols[:, 0]
            pen_up = pen_cols[:, 1]
        else:
            pen_up = np.zeros(len(coords), dtype=np.int64)
            pen_up[-1] = 1
    elif pen_up is None:
        pen_up = np.zeros(len(coords), dtype=np.int64)
        pen_up[-1] = 1

    pen_down = 1 - pen_up
    eos = np.zeros(len(coords), dtype=np.int64)
    eos[-1] = 1

    sketch = np.zeros((len(coords), 5), dtype=np.float32)
    sketch[:, 0:2] = deltas
    sketch[:, 2] = pen_down
    sketch[:, 3] = pen_up
    sketch[:, 4] = eos
    return sketch


class SketchDataset(Dataset):
    """Dataset that returns sketch-5 sequences and labels."""

    def __init__(self, samples: List[ManifestSample], max_retries: int = 5):
        self.samples = samples
        self.max_retries = max_retries

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        for attempt in range(self.max_retries):
            sample = self.samples[idx]
            try:
                arr = safe_np_load(sample.path)
                coords = extract_coords(arr)
                if coords is None:
                    raise ValueError("Invalid coords")
                sketch = coords_to_sketch5(coords)
                return torch.from_numpy(sketch), sample.label
            except Exception:
                idx = (idx + 1) % len(self.samples)
        raise RuntimeError("Failed to load sketch after retries")


def collate_sketch(batch: List[Tuple[torch.Tensor, int]]):
    sequences, labels = zip(*batch)
    lengths = torch.tensor([seq.shape[0] for seq in sequences], dtype=torch.long)
    max_len = max(lengths).item()

    padded = torch.zeros((len(sequences), max_len, 5), dtype=torch.float32)
    for i, seq in enumerate(sequences):
        padded[i, : seq.shape[0]] = seq
    labels = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels
