"""Sample sketches from a trained conditional LSTM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.generation.train_lstm import ConditionalLSTM
from src.utils.paths import ensure_dir
from src.utils.rasterize_from_npy import rasterize_coords


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from conditional LSTM")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to best_condlstm.pt")
    parser.add_argument("--out_dir", type=str, default="figures")
    parser.add_argument("--n_per_class", type=int, default=5)
    parser.add_argument("--T", type=int, default=100)
    parser.add_argument("--temp", type=float, default=0.7)
    parser.add_argument("--img_size", type=int, default=64)
    return parser.parse_args()


def sketch5_to_coords(sketch: np.ndarray) -> np.ndarray:
    xy = np.cumsum(sketch[:, :2], axis=0)
    pen_up = sketch[:, 3]
    coords = np.zeros((len(sketch), 3), dtype=np.float32)
    coords[:, :2] = xy
    coords[:, 2] = (pen_up > 0.5).astype(np.float32)
    return coords


def sample_class(model: ConditionalLSTM, label: int, T: int, temp: float, device: torch.device):
    model.eval()
    steps = []
    prev = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0]], device=device)
    label_tensor = torch.tensor([label], device=device)

    with torch.no_grad():
        for _ in range(T):
            inp = prev.unsqueeze(0)
            pred_xy, pred_pen = model(inp, label_tensor)
            pred_xy = pred_xy[:, -1]
            logits = pred_pen[:, -1] / max(temp, 1e-6)
            probs = torch.softmax(logits, dim=-1)
            pen_state = torch.multinomial(probs, 1).item()
            dxdy = pred_xy.squeeze(0)
            if temp > 0:
                dxdy = dxdy + temp * 0.05 * torch.randn_like(dxdy)

            pen_onehot = torch.zeros(3, device=device)
            pen_onehot[pen_state] = 1.0
            step = torch.cat([dxdy, pen_onehot], dim=0)
            steps.append(step.cpu().numpy())
            prev = step

            if pen_state == 2:
                break

    if not steps:
        steps = [np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)]
    return np.stack(steps)


def main() -> None:
    args = parse_args()
    ckpt = torch.load(args.ckpt, map_location="cpu")
    classes = ckpt["classes"]

    model = ConditionalLSTM(
        num_classes=len(classes),
        embed_dim=ckpt.get("embed_dim", 64),
        hidden_dim=ckpt.get("hidden_dim", 256),
    )
    model.load_state_dict(ckpt["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    out_root = ensure_dir(Path(args.out_dir)) / "gen_samples"

    for class_idx, class_name in enumerate(classes):
        class_dir = ensure_dir(out_root / class_name)
        for k in range(args.n_per_class):
            sketch = sample_class(model, class_idx, args.T, args.temp, device)
            coords = sketch5_to_coords(sketch)
            img = rasterize_coords(coords, size=args.img_size)
            img_np = (img.squeeze(0).numpy() * 255).astype(np.uint8)
            Image.fromarray(img_np, mode="L").save(class_dir / f"{class_name}_{k}.png")


if __name__ == "__main__":
    main()
