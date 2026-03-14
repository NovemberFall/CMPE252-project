
import os
import random
import shutil
from pathlib import Path

# Config
BASE_DIR = Path("dataset")
TRAIN_DIR = BASE_DIR / "train"
VAL_DIR = BASE_DIR / "val"
SPLIT_RATIO = 0.20  # 20% for validation

def splitDataset():
    if not TRAIN_DIR.exists():
        print(f"Error: {TRAIN_DIR} not found. Make sure your images are in dataset/train/")
        return

    # Categories: benign, malignant, normal
    categories = [d.name for d in TRAIN_DIR.iterdir() if d.is_dir()]

    for category in categories:
        # Create corresponding val folder
        target_val_dir = VAL_DIR / category
        target_val_dir.mkdir(parents=True, exist_ok=True)

        # Get all images in the train category
        src_dir = TRAIN_DIR / category
        images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Calculate how many to move
        num_to_move = int(len(images) * SPLIT_RATIO)
        to_move = random.sample(images, num_to_move)

        print(f"Moving {num_to_move} images from {category} to validation...")

        for img_name in to_move:
            shutil.move(src_dir / img_name, target_val_dir / img_name)

    print("\nDone! Your dataset is now split 80/20.")

if __name__ == "__main__":
    splitDataset()