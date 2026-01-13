from pathlib import Path
p = Path("src/recognition/train_from_coords.py")
txt = p.read_text()
txt = txt.replace("num_workers=2", "num_workers=0")
p.write_text(txt)
print("Patched num_workers to 0")
