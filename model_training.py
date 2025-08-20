import os
import random
import shutil
import yaml
from pathlib import Path

# Paths
dataset_dir = Path("/home/extra_space/akhilesh/severstal-steel-defect-detection")
train_img_dir = dataset_dir / "train_images"
label_dir = dataset_dir / "labels"

# Create split folders
split_dir = dataset_dir / "split"
split_train_img = split_dir / "images" / "train"
split_val_img = split_dir / "images" / "val"
split_train_lbl = split_dir / "labels" / "train"
split_val_lbl = split_dir / "labels" / "val"

for d in [split_train_img, split_val_img, split_train_lbl, split_val_lbl]:
    os.makedirs(d, exist_ok=True)

# Split train into 80% train / 20% val
images = [f for f in os.listdir(train_img_dir) if f.endswith(".jpg")]
random.shuffle(images)
val_size = int(0.2 * len(images))
val_images = images[:val_size]
train_images = images[val_size:]

def move_files(img_list, img_src, lbl_src, img_dst, lbl_dst):
    for img in img_list:
        base = Path(img).stem   # get filename without extension
        lbl_file = f"{base}.txt"

        # Copy image
        shutil.copy(img_src / img, img_dst / img)

        # Copy matching label
        lbl_path = lbl_src / lbl_file
        if lbl_path.exists():
            shutil.copy(lbl_path, lbl_dst / lbl_file)
        else:
            print(f"⚠️ Warning: No label found for {img}, skipping label copy.")

# Move images + labels
move_files(train_images, train_img_dir, label_dir, split_train_img, split_train_lbl)
move_files(val_images, train_img_dir, label_dir, split_val_img, split_val_lbl)

print(f"✅ Train images: {len(train_images)}, Val images: {len(val_images)}")

# Create data.yaml
data_yaml = {
    'train': str(split_train_img.resolve()),
    'val': str(split_val_img.resolve()),
    'nc': 4,   # number of classes (adjust if needed)
    'names': ['crazing', 'inclusion', 'pitted_surface', 'scratches']  # edit per dataset
}

with open(dataset_dir / "data.yaml", 'w') as f:
    yaml.dump(data_yaml, f, default_flow_style=False)

print("✅ data.yaml created at:", dataset_dir / "data.yaml")

# Training command
print("\nRun this command to train:")
print(f"python yolov5/segment/train.py --img 256 1600 --batch 16 --epochs 100 "
      f"--data {dataset_dir/'data.yaml'} --weights yolov5s-seg.pt --device 0")
