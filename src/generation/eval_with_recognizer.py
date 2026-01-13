"""Evaluate generated images with trained recognizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Dataset

from src.recognition.model import build_resnet18
from src.utils.paths import ensure_dir
from src.utils.plotting import plot_confusion_matrix


class GeneratedImageDataset(Dataset):
    def __init__(self, root: Path, classes: list[str], img_size: int = 64):
        self.samples = []
        self.classes = classes
        self.img_size = img_size
        for idx, class_name in enumerate(classes):
            class_dir = root / class_name
            if not class_dir.exists():
                continue
            for path in class_dir.glob("*.png"):
                self.samples.append((path, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("L").resize((self.img_size, self.img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr)[None, :, :]
        return tensor, label


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eval generated images with recognizer")
    parser.add_argument("--rec_ckpt", type=str, default=None, help="Path to best_recognizer.pt")
    parser.add_argument("--gen_root", type=str, default=None, help="Root folder of generated samples")
    parser.add_argument("--out_dir", type=str, default="outputs/eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path.cwd()

    rec_ckpt = Path(args.rec_ckpt) if args.rec_ckpt else repo_root / "outputs" / "recognition" / "best_recognizer.pt"
    if not rec_ckpt.exists():
        raise FileNotFoundError(f"Recognizer checkpoint not found: {rec_ckpt}")

    ckpt = torch.load(rec_ckpt, map_location="cpu")
    classes = ckpt["classes"]
    img_size = ckpt.get("img_size", 64)

    gen_root = Path(args.gen_root) if args.gen_root else repo_root / "figures" / "gen_samples"
    if not gen_root.exists():
        raise FileNotFoundError(f"Generated samples root not found: {gen_root}")

    dataset = GeneratedImageDataset(gen_root, classes, img_size=img_size)
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(classes))
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()

    preds = []
    targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            batch_preds = torch.argmax(logits, dim=1).cpu().numpy()
            preds.append(batch_preds)
            targets.append(np.array(labels))

    preds = np.concatenate(preds) if preds else np.array([])
    targets = np.concatenate(targets) if targets else np.array([])
    acc = float((preds == targets).mean()) if len(preds) else 0.0

    per_class = {}
    for i, name in enumerate(classes):
        mask = targets == i
        per_class[name] = float((preds[mask] == i).mean()) if mask.any() else 0.0

    cm = confusion_matrix(targets, preds, labels=list(range(len(classes)))) if len(preds) else np.zeros((len(classes), len(classes)))

    out_dir = ensure_dir(Path(args.out_dir))
    metrics = {
        "accuracy": acc,
        "per_class_accuracy": per_class,
        "confusion_matrix": cm.tolist(),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_confusion_matrix(cm, classes, out_dir / "confusion_matrix.png")

    print(f"Generated sample accuracy: {acc:.4f}")


if __name__ == "__main__":
    main()
