"""Train a conditional LSTM generator."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.generation.dataset import SketchDataset, collate_sketch, load_manifest
from src.utils.paths import ensure_dir
from src.utils.seed import set_seed


class ConditionalLSTM(nn.Module):
    def __init__(self, num_classes: int, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.lstm = nn.LSTM(input_size=5 + embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.fc_xy = nn.Linear(hidden_dim, 2)
        self.fc_pen = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor, labels: torch.Tensor):
        if x.dim() == 2:
            x = x.unsqueeze(0)
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        labels = labels.view(-1)
        if labels.numel() == 1 and x.size(0) > 1:
            labels = labels.expand(x.size(0))
        emb = self.embedding(labels)
        emb = emb.unsqueeze(1).expand(x.size(0), x.size(1), -1)
        lstm_in = torch.cat([x, emb], dim=-1)
        out, _ = self.lstm(lstm_in)
        xy = self.fc_xy(out)
        pen = self.fc_pen(out)
        return xy, pen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train conditional LSTM")
    parser.add_argument("--rec_out", type=str, required=True, help="Path to recognition manifest dir")
    parser.add_argument("--out_dir", type=str, default="outputs/generation")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def masked_loss(pred_xy, target_xy, pred_pen, target_pen, lengths):
    batch_size, seq_len, _ = pred_xy.shape
    mask = torch.arange(seq_len, device=lengths.device)[None, :] < lengths[:, None]
    mask = mask.float()

    mse = ((pred_xy - target_xy) ** 2).sum(dim=-1) * mask
    mse = mse.sum() / mask.sum().clamp(min=1.0)

    ce = nn.functional.cross_entropy(
        pred_pen.reshape(-1, 3),
        target_pen.reshape(-1),
        reduction="none",
    ).reshape(batch_size, seq_len)
    ce = (ce * mask).sum() / mask.sum().clamp(min=1.0)
    return mse + ce


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = ensure_dir(Path(args.out_dir))
    manifest_dir = Path(args.rec_out)

    train_manifest = manifest_dir / "manifest_train.csv"
    val_manifest = manifest_dir / "manifest_val.csv"
    classes_path = manifest_dir / "classes.txt"

    if not train_manifest.exists():
        raise FileNotFoundError(f"Missing manifest: {train_manifest}")

    classes = classes_path.read_text().strip().splitlines()

    train_samples = load_manifest(train_manifest)
    val_samples = load_manifest(val_manifest)

    train_dataset = SketchDataset(train_samples)
    val_dataset = SketchDataset(val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=collate_sketch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch,
        shuffle=False,
        collate_fn=collate_sketch,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConditionalLSTM(num_classes=len(classes)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_path = out_dir / "best_condlstm.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for batch, lengths, labels in train_loader:
            batch = batch.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            inp = batch[:, :-1, :]
            target = batch[:, 1:, :]
            target_xy = target[:, :, :2]
            target_pen = torch.argmax(target[:, :, 2:], dim=-1)
            lengths_adj = torch.clamp(lengths - 1, min=1)

            optimizer.zero_grad()
            pred_xy, pred_pen = model(inp, labels)
            loss = masked_loss(pred_xy, target_xy, pred_pen, target_pen, lengths_adj)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        train_loss = train_loss / max(len(train_dataset), 1)
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for batch, lengths, labels in val_loader:
                batch = batch.to(device)
                lengths = lengths.to(device)
                labels = labels.to(device)
                inp = batch[:, :-1, :]
                target = batch[:, 1:, :]
                target_xy = target[:, :, :2]
                target_pen = torch.argmax(target[:, :, 2:], dim=-1)
                lengths_adj = torch.clamp(lengths - 1, min=1)
                pred_xy, pred_pen = model(inp, labels)
                loss = masked_loss(pred_xy, target_xy, pred_pen, target_pen, lengths_adj)
                val_loss += loss.item() * batch.size(0)
        val_loss = val_loss / max(len(val_dataset), 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "classes": classes,
                    "embed_dim": model.embedding.embedding_dim,
                    "hidden_dim": model.lstm.hidden_size,
                },
                best_path,
            )

    history_path = out_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
