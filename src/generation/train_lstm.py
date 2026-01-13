import os, json, time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.generation.dataset import SketchCoordDataset

class CondLSTM(nn.Module):
    def __init__(self, n_classes, emb_dim=32, hid=256):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.lstm = nn.LSTM(input_size=5 + emb_dim, hidden_size=hid, batch_first=True)
        self.out_xy = nn.Linear(hid, 2)
        self.out_pen = nn.Linear(hid, 3)  # cont / stroke_end / eos

    def forward(self, x, y):
        e = self.emb(y)[:, None, :].expand(x.size(0), x.size(1), -1)
        h, _ = self.lstm(torch.cat([x, e], dim=-1))
        return self.out_xy(h), self.out_pen(h)

def main():
    REC_OUT = Path(os.environ.get("REC_OUT", "/content/drive/MyDrive/cs3308_quickdraw/outputs/recognition_from_coords"))
    OUT_DIR = Path(os.environ.get("OUT_DIR", "/content/drive/MyDrive/cs3308_quickdraw/outputs/generation"))
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    classes = (REC_OUT / "classes.txt").read_text().splitlines()
    n_classes = len(classes)

    train_csv = str(REC_OUT / "manifest_train.csv")
    val_csv   = str(REC_OUT / "manifest_val.csv")

    batch  = int(os.environ.get("BATCH", "64"))
    epochs = int(os.environ.get("EPOCHS", "1"))
    lr     = float(os.environ.get("LR", "1e-3"))

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Device:", dev, "| classes:", n_classes)

    tr = SketchCoordDataset(train_csv)
    va = SketchCoordDataset(val_csv)
    tr_dl = DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0)
    va_dl = DataLoader(va, batch_size=batch, shuffle=False, num_workers=0)

    model = CondLSTM(n_classes=n_classes).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()
    ce  = nn.CrossEntropyLoss()

    history = {"epochs": []}
    best = 1e9

    for ep in range(1, epochs+1):
        t0 = time.time()
        model.train()
        tr_loss, n = 0.0, 0

        for x, y in tr_dl:
            x, y = x.to(dev), y.to(dev)
            xin = x[:, :-1, :]
            tgt_xy = x[:, 1:, :2]
            tgt_pen = x[:, 1:, 2:].argmax(dim=-1)

            pred_xy, pred_pen = model(xin, y)
            loss = mse(pred_xy, tgt_xy) + ce(pred_pen.reshape(-1,3), tgt_pen.reshape(-1))

            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            tr_loss += loss.item()
            n += 1

        tr_loss /= max(n, 1)

        model.eval()
        va_loss, m = 0.0, 0
        with torch.no_grad():
            for x, y in va_dl:
                x, y = x.to(dev), y.to(dev)
                xin = x[:, :-1, :]
                tgt_xy = x[:, 1:, :2]
                tgt_pen = x[:, 1:, 2:].argmax(dim=-1)
                pred_xy, pred_pen = model(xin, y)
                loss = mse(pred_xy, tgt_xy) + ce(pred_pen.reshape(-1,3), tgt_pen.reshape(-1))
                va_loss += loss.item()
                m += 1
        va_loss /= max(m, 1)

        dt = time.time() - t0
        print(f"Epoch {ep}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f} time={dt:.1f}s")

        history["epochs"].append({"epoch": ep, "train_loss": tr_loss, "val_loss": va_loss})
        (OUT_DIR/"history.json").write_text(json.dumps(history, indent=2))

        if va_loss < best:
            best = va_loss
            torch.save({"model": model.state_dict(), "classes": classes}, OUT_DIR/"best_condlstm.pt")

    print("Saved:", OUT_DIR/"best_condlstm.pt")

if __name__ == "__main__":
    main()
