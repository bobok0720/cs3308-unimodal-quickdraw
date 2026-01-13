import csv, random
import numpy as np
import torch
from torch.utils.data import Dataset

def safe_np_load(path: str):
    try:
        return np.load(path, allow_pickle=True)
    except UnicodeError:
        return np.load(path, allow_pickle=True, encoding="latin1")

def extract_T4(obj):
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.ndim == 0:
        obj = obj.item()

    if isinstance(obj, np.ndarray) and obj.dtype != object:
        if obj.ndim == 2 and obj.shape[1] == 4:
            return obj
        if obj.ndim == 3 and obj.shape[-1] == 4:
            return obj[0]
        raise ValueError(f"Numeric array not (T,4): shape={obj.shape}, dtype={obj.dtype}")

    if isinstance(obj, (list, tuple)):
        for it in obj:
            try:
                return extract_T4(it)
            except Exception:
                pass
        raise ValueError("No (T,4) inside list/tuple")

    if isinstance(obj, np.ndarray) and obj.dtype == object:
        for i in range(obj.size):
            try:
                return extract_T4(obj.flat[i])
            except Exception:
                pass
        raise ValueError(f"No (T,4) inside object ndarray: shape={obj.shape}")

    raise ValueError(f"Unsupported type: {type(obj)}")

def to_sketch5(a4: np.ndarray):
    a4 = np.asarray(a4)
    assert a4.ndim == 2 and a4.shape[1] == 4, f"Expected (T,4), got {a4.shape}"

    xy = a4[:, :2].astype(np.float32)
    pen2 = a4[:, 2:4].astype(np.float32)

    if float(np.max(np.abs(xy))) > 50.0:
        xy = np.diff(xy, axis=0, prepend=xy[:1])

    pen2 = (pen2 > 0.5).astype(np.float32)
    last = pen2[-1]
    eos_col = int(np.argmax(last)) if last.sum() >= 1 else 1

    eos = pen2[:, eos_col]
    stroke_end = pen2[:, 1 - eos_col]
    cont = 1.0 - np.clip(stroke_end + eos, 0.0, 1.0)

    return np.concatenate([xy, cont[:, None], stroke_end[:, None], eos[:, None]], axis=1).astype(np.float32)

class SketchCoordDataset(Dataset):
    def __init__(self, manifest_csv: str, max_retries: int = 10, seed: int = 0):
        self.rows = []
        with open(manifest_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append((row["path"], int(row["label"])))
        self.max_retries = max_retries
        self.rng = random.Random(seed)

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        for _ in range(self.max_retries):
            path, y = self.rows[idx]
            try:
                raw = safe_np_load(path)
                a4 = extract_T4(raw)
                x = to_sketch5(a4)
                return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)
            except Exception:
                idx = self.rng.randrange(len(self.rows))

        x = np.zeros((100,5), dtype=np.float32)
        x[-1, 4] = 1.0
        return torch.from_numpy(x), torch.tensor(0, dtype=torch.long)
