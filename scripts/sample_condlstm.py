import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw
import torch
import torch.nn as nn

GEN_OUT = Path("/content/drive/MyDrive/cs3308_quickdraw/outputs/generation")
FIG_OUT = Path("/content/drive/MyDrive/cs3308_quickdraw/figures/gen_samples")
FIG_OUT.mkdir(parents=True, exist_ok=True)

class CondLSTM(nn.Module):
    def __init__(self, n_classes, emb_dim=32, hid=256):
        super().__init__()
        self.emb = nn.Embedding(n_classes, emb_dim)
        self.lstm = nn.LSTM(input_size=5 + emb_dim, hidden_size=hid, batch_first=True)
        self.out_xy = nn.Linear(hid, 2)
        self.out_pen = nn.Linear(hid, 3)  # cont / stroke_end / eos

    def forward(self, x, y, h=None):
        e = self.emb(y)[:, None, :].expand(x.size(0), x.size(1), -1)
        out, h2 = self.lstm(torch.cat([x, e], dim=-1), h)
        return self.out_xy(out), self.out_pen(out), h2

def draw_sketch5(seq5, size=256, padding=10, width=2):
    seq5 = np.asarray(seq5, dtype=np.float32)
    dxy = seq5[:, :2]
    stroke_end = seq5[:, 3] > 0.5
    eos = seq5[:, 4] > 0.5
    xy = np.cumsum(dxy, axis=0)

    xs, ys = xy[:,0], xy[:,1]
    minx, maxx = float(xs.min()), float(xs.max())
    miny, maxy = float(ys.min()), float(ys.max())
    w = max(maxx - minx, 1e-6)
    h = max(maxy - miny, 1e-6)
    scale = (size - 2*padding) / max(w, h)

    def tx(x): return (x - minx) * scale + padding
    def ty(y): return (y - miny) * scale + padding

    img = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    cur = []
    for i in range(len(xy)):
        cur.append((tx(xy[i,0]), ty(xy[i,1])))
        if stroke_end[i] or eos[i] or i == len(xy)-1:
            if len(cur) >= 2: draw.line(cur, fill=255, width=width)
            elif len(cur) == 1: draw.point(cur[0], fill=255)
            cur = []
        if eos[i]: break
    return img

@torch.no_grad()
def sample(model, cls_id, T=100, temp=0.9, dev="cpu"):
    y = torch.tensor([cls_id], device=dev)
    x = torch.zeros(1, 1, 5, device=dev)  # start token
    h = None
    seq = []

    for _ in range(T):
        pred_xy, pred_pen, h = model(x, y, h)
        pred_xy = pred_xy[:, -1, :]                 # (1,2)
        logits = pred_pen[:, -1, :] / temp          # (1,3)
        pen_id = torch.distributions.Categorical(logits=logits).sample().item()

        step = torch.zeros(1, 5, device=dev)
        step[0, :2] = pred_xy[0]
        step[0, 2 + pen_id] = 1.0  # one-hot: cont/stroke_end/eos
        seq.append(step.squeeze(0).cpu().numpy())

        if pen_id == 2:
            break
        x = step.view(1, 1, 5)

    # pad to fixed length for convenience
    if len(seq) < T:
        pad = np.zeros((T - len(seq), 5), dtype=np.float32)
        pad[-1, 4] = 1.0
        seq = np.vstack([np.asarray(seq, dtype=np.float32), pad])
    else:
        seq = np.asarray(seq[:T], dtype=np.float32)

    return seq

ckpt = torch.load(GEN_OUT / "best_condlstm.pt", map_location="cpu")
classes = ckpt["classes"]
n_classes = len(classes)

dev = "cuda" if torch.cuda.is_available() else "cpu"
model = CondLSTM(n_classes=n_classes).to(dev)
model.load_state_dict(ckpt["model"])
model.eval()

# generate
for cid, cname in enumerate(classes):
    out_dir = FIG_OUT / cname
    out_dir.mkdir(parents=True, exist_ok=True)
    for k in range(16):
        s = sample(model, cid, T=100, temp=0.9, dev=dev)
        img = draw_sketch5(s)
        img.save(out_dir / f"{cname}_{k}.png")

print("Saved to:", FIG_OUT)
