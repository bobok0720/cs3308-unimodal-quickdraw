"""Evaluation helpers for recognition."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import f1_score


def evaluate(model: torch.nn.Module, loader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(labels.cpu().numpy())
    preds = np.concatenate(all_preds) if all_preds else np.array([])
    targets = np.concatenate(all_targets) if all_targets else np.array([])
    acc = float((preds == targets).mean()) if len(preds) else 0.0
    macro_f1 = float(f1_score(targets, preds, average="macro")) if len(preds) else 0.0
    return acc, macro_f1
