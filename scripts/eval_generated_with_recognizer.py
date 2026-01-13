import json
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms

# Paths
REC_OUT = Path("/content/drive/MyDrive/cs3308_quickdraw/outputs/recognition_from_coords")
GEN_DIR = Path("/content/drive/MyDrive/cs3308_quickdraw/figures/gen_samples")
OUT_DIR = Path("/content/drive/MyDrive/cs3308_quickdraw/outputs/generation_eval")
FIG_DIR = Path("/content/drive/MyDrive/cs3308_quickdraw/figures")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

def find_rec_ckpt(rec_out: Path) -> Path:
    pts = sorted(rec_out.glob("*.pt"))
    if not pts:
        raise FileNotFoundError(f"No .pt found in {rec_out}")

    # Prefer best* then coords* then newest
    best = [p for p in pts if "best" in p.name.lower()]
    if best:
        pts = best
    coords = [p for p in pts if "coord" in p.name.lower()]
    if coords:
        pts = coords
    pts = sorted(pts, key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0]

def build_resnet18_gray(n_classes: int) -> nn.Module:
    m = models.resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, n_classes)
    return m

def load_state_into_model(model: nn.Module, ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    # common patterns:
    #  - state_dict directly
    #  - {"model": state_dict, ...}
    if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], dict):
        sd = ckpt["model"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        sd = ckpt["state_dict"]
    elif isinstance(ckpt, dict):
        # might already be a state dict-like dict
        sd = ckpt
    else:
        raise ValueError(f"Unsupported checkpoint format: {type(ckpt)}")

    missing, unexpected = model.load_state_dict(sd, strict=False)
    return ckpt, missing, unexpected

def main():
    classes = REC_OUT.joinpath("classes.txt").read_text().splitlines()
    n_classes = len(classes)
    class_to_id = {c: i for i, c in enumerate(classes)}

    ckpt_path = find_rec_ckpt(REC_OUT)
    print("Using recognizer checkpoint:", ckpt_path)

    model = build_resnet18_gray(n_classes)
    ckpt, missing, unexpected = load_state_into_model(model, ckpt_path)
    if missing:
        print("Missing keys (ok if none/harmless):", missing[:10], "..." if len(missing) > 10 else "")
    if unexpected:
        print("Unexpected keys (ok if none/harmless):", unexpected[:10], "..." if len(unexpected) > 10 else "")

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(dev).eval()

    tfm = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),  # (1,H,W) in [0,1] for L-mode
    ])

    # Collect images
    rows = []
    for class_dir in sorted(GEN_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        cname = class_dir.name
        if cname not in class_to_id:
            continue
        y_true = class_to_id[cname]
        for img_path in sorted(class_dir.glob("*.png")):
            rows.append((img_path, y_true, cname))

    if not rows:
        raise FileNotFoundError(f"No PNGs found under {GEN_DIR}")

    # Evaluate
    conf = np.zeros((n_classes, n_classes), dtype=np.int64)
    correct = 0
    total = 0

    with torch.no_grad():
        for img_path, y_true, cname in rows:
            img = Image.open(img_path).convert("L")
            x = tfm(img).unsqueeze(0).to(dev)  # (1,1,64,64)
            logits = model(x)
            y_pred = int(torch.argmax(logits, dim=1).item())
            conf[y_true, y_pred] += 1
            total += 1
            correct += int(y_pred == y_true)

    acc = correct / total
    per_class_acc = {}
    for i, cname in enumerate(classes):
        denom = int(conf[i].sum())
        per_class_acc[cname] = (int(conf[i, i]) / denom) if denom else None

    out = {
        "recognizer_ckpt": str(ckpt_path),
        "n_classes": n_classes,
        "total_samples": total,
        "accuracy": acc,
        "per_class_accuracy": per_class_acc,
        "classes": classes,
        "confusion_matrix": conf.tolist(),
    }

    (OUT_DIR / "gen_eval_metrics.json").write_text(json.dumps(out, indent=2))
    print("Saved metrics:", OUT_DIR / "gen_eval_metrics.json")

    # Plot confusion matrix
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 6))
    plt.imshow(conf, aspect="auto")
    plt.title("Generated Samples → Recognition Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(range(n_classes), classes, rotation=45, ha="right", fontsize=8)
    plt.yticks(range(n_classes), classes, fontsize=8)
    plt.tight_layout()
    fig_path = FIG_DIR / "gen_confusion_matrix.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()
    print("Saved figure:", fig_path)

if __name__ == "__main__":
    main()
