"""Train a recognition model from coordinate files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader

from src.recognition.dataset import CoordRasterDataset, load_manifest
from src.recognition.model import build_resnet18
from src.recognition.eval import evaluate
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train recognizer from coords")
    parser.add_argument("--coord_root", type=str, required=True, help="Coordinate root")
    parser.add_argument("--out_dir", type=str, default="outputs/recognition")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = ensure_dir(Path(args.out_dir))
    manifest_dir = Path(args.coord_root)

    train_manifest = manifest_dir / "manifest_train.csv"
    val_manifest = manifest_dir / "manifest_val.csv"
    test_manifest = manifest_dir / "manifest_test.csv"
    classes_path = manifest_dir / "classes.txt"

    if not train_manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {train_manifest}")

    classes = classes_path.read_text().strip().splitlines()

    train_samples = load_manifest(train_manifest)
    val_samples = load_manifest(val_manifest)
    test_samples = load_manifest(test_manifest)

    train_dataset = CoordRasterDataset(train_samples, img_size=args.img_size)
    val_dataset = CoordRasterDataset(val_samples, img_size=args.img_size)
    test_dataset = CoordRasterDataset(test_samples, img_size=args.img_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet18(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_acc": [], "val_macro_f1": []}
    best_f1 = -1.0
    best_path = out_dir / "best_recognizer.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / max(len(train_dataset), 1)
        val_acc, val_f1 = evaluate(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val acc {val_acc:.4f} | val macro-F1 {val_f1:.4f}"
        )

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "img_size": args.img_size,
                },
                best_path,
            )

    out_dir.mkdir(parents=True, exist_ok=True)
    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))

    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])

    test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"Test acc {test_acc:.4f} | Test macro-F1 {test_f1:.4f}")


if __name__ == "__main__":
    main()
