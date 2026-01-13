import json
from pathlib import Path
import matplotlib.pyplot as plt

OUT_DIR = Path("/content/drive/MyDrive/cs3308_quickdraw/outputs/recognition_from_coords")
FIG_DIR = Path("/content/drive/MyDrive/cs3308_quickdraw/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

hist = json.loads((OUT_DIR/"history.json").read_text())
epochs = [e["epoch"] for e in hist["epochs"]]
train_loss = [e["train_loss"] for e in hist["epochs"]]
val_acc = [e["val_acc"] for e in hist["epochs"]]
val_f1 = [e["val_macro_f1"] for e in hist["epochs"]]

plt.figure()
plt.plot(epochs, train_loss)
plt.xlabel("Epoch"); plt.ylabel("Train Loss"); plt.title("Recognition Train Loss")
plt.savefig(FIG_DIR/"rec_train_loss.png", dpi=200); plt.close()

plt.figure()
plt.plot(epochs, val_acc)
plt.xlabel("Epoch"); plt.ylabel("Val Accuracy"); plt.title("Recognition Validation Accuracy")
plt.savefig(FIG_DIR/"rec_val_acc.png", dpi=200); plt.close()

plt.figure()
plt.plot(epochs, val_f1)
plt.xlabel("Epoch"); plt.ylabel("Val Macro-F1"); plt.title("Recognition Validation Macro-F1")
plt.savefig(FIG_DIR/"rec_val_macroF1.png", dpi=200); plt.close()

print("Saved:", FIG_DIR)
