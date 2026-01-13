"""Utilities to load coordinate .npy files and rasterize them."""

from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw


def safe_np_load(path: str):
    """Robust np.load for object/pickled .npy created in older Python."""

    try:
        return np.load(path, allow_pickle=True)
    except UnicodeError:
        return np.load(path, allow_pickle=True, encoding="latin1")
    except Exception:
        return np.load(path, allow_pickle=False)


def extract_coords(obj) -> Optional[np.ndarray]:
    """Extract a (T,4) or (T,3) coordinate array from a variety of wrappers."""

    if isinstance(obj, np.ndarray):
        if obj.dtype == object and obj.size == 1:
            try:
                return extract_coords(obj.item())
            except Exception:
                pass
        if obj.ndim == 3 and obj.shape[0] == 1:
            return extract_coords(obj[0])
        if obj.ndim == 2 and obj.shape[1] >= 2:
            return obj
    if isinstance(obj, (list, tuple)):
        if len(obj) == 1:
            return extract_coords(obj[0])
        try:
            arr = np.asarray(obj)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                return arr
        except Exception:
            pass
    return None


def _coords_to_strokes(coords: np.ndarray) -> List[List[Tuple[float, float]]]:
    coords = np.asarray(coords)
    if coords.ndim != 2 or coords.shape[1] < 2:
        return []

    xy = coords[:, :2].astype(np.float32)
    max_abs = float(np.max(np.abs(xy))) if xy.size else 0.0
    if max_abs <= 20.0:
        xy = np.cumsum(xy, axis=0)

    pen = None
    if coords.shape[1] >= 3:
        pen = coords[:, 2].astype(np.int32)

    strokes: List[List[Tuple[float, float]]] = []
    current: List[Tuple[float, float]] = []
    for i in range(len(xy)):
        current.append((float(xy[i, 0]), float(xy[i, 1])))
        cut = False
        if pen is not None and pen[i] != 0:
            cut = True
        if cut and current:
            strokes.append(current)
            current = []
    if current:
        strokes.append(current)
    return strokes


def rasterize_coords(coords: np.ndarray, size: int = 64, padding: int = 5, stroke_width: int = 2) -> torch.Tensor:
    """Rasterize coordinates into a 1xHxW torch tensor in [0,1]."""

    strokes = _coords_to_strokes(coords)
    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    pts_all = [p for stroke in strokes for p in stroke]
    if not pts_all:
        out = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(out)[None, :, :]

    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)

    scale = (size - 2 * padding) / max(w, h)

    def tx(x: float) -> float:
        return (x - minx) * scale + padding

    def ty(y: float) -> float:
        return (y - miny) * scale + padding

    for stroke in strokes:
        if len(stroke) == 1:
            x, y = stroke[0]
            draw.point((tx(x), ty(y)), fill=255)
        else:
            line = [(tx(x), ty(y)) for x, y in stroke]
            draw.line(line, fill=255, width=stroke_width)

    out = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(out)[None, :, :]


def rasterize_npy(npy_path: str, size: int = 64, padding: int = 5, stroke_width: int = 2) -> torch.Tensor:
    arr = safe_np_load(npy_path)
    coords = extract_coords(arr)
    if coords is None:
        coords = np.zeros((1, 2), dtype=np.float32)
    return rasterize_coords(coords, size=size, padding=padding, stroke_width=stroke_width)
