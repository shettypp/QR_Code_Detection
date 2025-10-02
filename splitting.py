from pathlib import Path
import random, shutil

# ===========================
# Train/Val Splitter (flat labels folder)
# ===========================

BASE = Path("QR_Dataset")
IMAGES = BASE / "train_images"
LABELS = BASE / "labels"    # 👈 all labels here
DEST = BASE / "split2"
TRAIN_IMG = DEST / "images" / "train"
VAL_IMG   = DEST / "images" / "val"
TRAIN_LBL = DEST / "labels" / "train"
VAL_LBL   = DEST / "labels" / "val"

# Cleanup old
if DEST.exists():
    shutil.rmtree(DEST)
for d in [TRAIN_IMG, VAL_IMG, TRAIN_LBL, VAL_LBL]:
    d.mkdir(parents=True, exist_ok=True)

# Collect all images
exts = {".jpg", ".jpeg", ".png"}
images = [p for p in IMAGES.iterdir() if p.suffix.lower() in exts]

# Shuffle + split
random.seed(42)
random.shuffle(images)
split_idx = int(0.8 * len(images))
train_files, val_files = images[:split_idx], images[split_idx:]

# Copy function
def copy_pair(img_path: Path, img_target: Path, lbl_target: Path):
    lbl_path = LABELS / f"{img_path.stem}.txt"
    shutil.copy2(img_path, img_target / img_path.name)
    if lbl_path.exists():
        shutil.copy2(lbl_path, lbl_target / lbl_path.name)
    else:
        print(f"⚠️ No label for {img_path.name}")

# Copy train/val sets
for f in train_files: copy_pair(f, TRAIN_IMG, TRAIN_LBL)
for f in val_files: copy_pair(f, VAL_IMG, VAL_LBL)

print("✅ Split complete!")
print(f"Training images: {len(list(TRAIN_IMG.glob('*.jpg')))}")
print(f"Validation images: {len(list(VAL_IMG.glob('*.jpg')))}")
