import numpy as np
from PIL import Image, ImageDraw

def safe_np_load(path: str):
    """
    Robust np.load for object/pickled .npy created in older Python.
    """
    try:
        return np.load(path, allow_pickle=True)
    except UnicodeError:
        # Common fix for Python2 pickles read in Python3
        return np.load(path, allow_pickle=True, encoding="latin1")
    except Exception:
        # last resort: try without pickle (numeric arrays)
        return np.load(path, allow_pickle=False)

def _as_drawing_from_xy_pen(xy_pen: np.ndarray):
    arr = np.asarray(xy_pen)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Expected (N,>=2) array")

    xy = arr[:, :2].astype(np.float32)

    # Heuristic: treat small magnitude as deltas -> cum-sum
    max_abs = float(np.max(np.abs(xy))) if xy.size else 0.0
    if max_abs <= 20.0:
        xy = np.cumsum(xy, axis=0)

    pen = arr[:, 2].astype(np.int32) if arr.shape[1] >= 3 else None

    strokes = []
    cur = []
    for i in range(len(xy)):
        cur.append((float(xy[i, 0]), float(xy[i, 1])))
        cut = False
        if pen is not None and pen[i] != 0:
            cut = True
        if cut and cur:
            strokes.append(cur)
            cur = []
    if cur:
        strokes.append(cur)
    return strokes

def _as_drawing_object(obj):
    strokes = []
    if isinstance(obj, (list, tuple)):
        for s in obj:
            s = np.asarray(s)
            if s.ndim == 2 and s.shape[1] >= 2:
                strokes.append([(float(x), float(y)) for x, y in s[:, :2]])
            elif s.ndim == 2 and s.shape[0] == 2:
                x = s[0].tolist()
                y = s[1].tolist()
                strokes.append([(float(xi), float(yi)) for xi, yi in zip(x, y)])
    return strokes

def rasterize_npy(npy_path: str, size=64, padding=5, stroke_width=2):
    arr = safe_np_load(npy_path)

    # unwrap object array (shape (1,))
    if isinstance(arr, np.ndarray) and arr.dtype == object and arr.size == 1:
        try:
            arr = arr.item()
        except Exception:
            pass

    strokes = []
    try:
        if isinstance(arr, np.ndarray) and arr.dtype != object and arr.ndim == 2:
            strokes = _as_drawing_from_xy_pen(arr)
        else:
            strokes = _as_drawing_object(arr)
    except Exception:
        strokes = []

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)

    pts_all = [p for stroke in strokes for p in stroke]
    if not pts_all:
        out = np.asarray(img, dtype=np.float32) / 255.0
        return out[None, :, :]

    xs = [p[0] for p in pts_all]
    ys = [p[1] for p in pts_all]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)

    scale = (size - 2 * padding) / max(w, h)

    def tx(x): return (x - minx) * scale + padding
    def ty(y): return (y - miny) * scale + padding

    for stroke in strokes:
        if len(stroke) == 1:
            x, y = stroke[0]
            draw.point((tx(x), ty(y)), fill=255)
        else:
            line = [(tx(x), ty(y)) for x, y in stroke]
            draw.line(line, fill=255, width=stroke_width)

    out = np.asarray(img, dtype=np.float32) / 255.0
    return out[None, :, :]
