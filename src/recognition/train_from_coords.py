import os, csv, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18
from sklearn.metrics import f1_score, confusion_matrix
from src.utils.rasterize_from_npy import rasterize_npy

def get_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class CoordRasterDataset(Dataset):
    def __init__(self, manifest_csv: str, img_size=64):
        self.rows = []
        with open(manifest_csv, "r", newline="") as f:
            r = csv.DictReader(f)
            for row in r:
                self.rows.append((row["path"], int(row["label"])))
        self.img_size = img_size

    def __len__(self): return len(self.rows)

    def __getitem__(self, idx):
        path, y = self.rows[idx]
        x = rasterize_npy(path, size=self.img_size)
        x = torch.from_numpy(x)  # (1,H,W)
        y = torch.tensor(y, dtype=torch.long)
        return x, y

def build_model(num_classes):
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m

@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(dev)
        logits = model(x)
        p = logits.argmax(1).cpu().tolist()
        ys.extend(y.tolist())
        ps.extend(p)
    acc = sum(int(a==b) for a,b in zip(ys, ps)) / max(len(ys), 1)
    macro_f1 = f1_score(ys, ps, average="macro")
    cm = confusion_matrix(ys, ps)
    return acc, macro_f1, cm

def main():
    OUT_DIR = Path(os.environ.get("OUT_DIR", "/content/drive/MyDrive/cs3308_quickdraw/outputs/recognition_from_coords"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    img_size   = int(os.environ.get("IMG_SIZE", "64"))
    batch_size = int(os.environ.get("BATCH", "128"))
    epochs     = int(os.environ.get("EPOCHS", "2"))
    lr         = float(os.environ.get("LR", "1e-3"))

    train_csv = OUT_DIR / "manifest_train.csv"
    val_csv   = OUT_DIR / "manifest_val.csv"
    test_csv  = OUT_DIR / "manifest_test.csv"

    assert train_csv.exists(), f"Missing {train_csv}. Run build_coord_manifest.py first."
    assert val_csv.exists(), f"Missing {val_csv}. Run build_coord_manifest.py first."
    assert test_csv.exists(), f"Missing {test_csv}. Run build_coord_manifest.py first."

    train_ds = CoordRasterDataset(str(train_csv), img_size=img_size)
    val_ds   = CoordRasterDataset(str(val_csv), img_size=img_size)
    test_ds  = CoordRasterDataset(str(test_csv), img_size=img_size)

    # num_classes from classes.txt
    classes = (OUT_DIR / "classes.txt").read_text(encoding="utf-8").splitlines()
    num_classes = len(classes)

    dev = get_device()
    print("Device:", dev, "| Classes:", num_classes)
    print("Sizes:", len(train_ds), len(val_ds), len(test_ds))

    # Drive is slow for many small files; keep workers modest
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)
    test_dl  = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=False)

    model = build_model(num_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    history = {"config": {"img_size": img_size, "batch": batch_size, "epochs": epochs, "lr": lr},
               "epochs": []}
    best_val_f1 = -1.0

    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0

        for x, y in train_dl:
            x, y = x.to(dev), y.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
            seen += x.size(0)

        train_loss = running / max(seen, 1)
        val_acc, val_f1, _ = evaluate(model, val_dl, dev)
        dt = time.time() - t0
        print(f"Epoch {ep}: train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_macroF1={val_f1:.4f} time={dt:.1f}s")

        history["epochs"].append({"epoch": ep, "train_loss": train_loss, "val_acc": val_acc, "val_macro_f1": val_f1, "seconds": dt})
        (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2))

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({"model": model.state_dict(), "classes": classes, "img_size": img_size},
                       OUT_DIR / "best_resnet18_coords.pt")

    ckpt = torch.load(OUT_DIR / "best_resnet18_coords.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])
    test_acc, test_f1, cm = evaluate(model, test_dl, dev)
    print(f"TEST: acc={test_acc:.4f} macroF1={test_f1:.4f}")

    torch.save(cm, OUT_DIR / "confusion_matrix.pt")

    cm2 = cm.copy().astype(np.int64)
    np.fill_diagonal(cm2, 0)
    flat = cm2.flatten()
    top_idx = flat.argsort()[::-1][:30]
    top_pairs = []
    for idx in top_idx:
        v = flat[idx]
        if v == 0: break
        i = idx // cm2.shape[1]
        j = idx % cm2.shape[1]
        top_pairs.append({"true": classes[i], "pred": classes[j], "count": int(v)})
    (OUT_DIR / "top_confusions.json").write_text(json.dumps(top_pairs, indent=2))

if __name__ == "__main__":
    main()
