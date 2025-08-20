import os
import pandas as pd
from PIL import Image
import numpy as np
import cv2

# Paths
csv_file = "/home/extra_space/akhilesh/severstal-steel-defect-detection/train.csv"
images_dir = "/home/extra_space/akhilesh/severstal-steel-defect-detection/train_images"
labels_dir = "/home/akhilesh/yolov5_new/yolov5/Eric_robotics_task/labels/"

os.makedirs(labels_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(csv_file)

def rle_decode(mask_rle, shape):
    """
    mask_rle: run-length as string (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 = mask, 0 = background
    """
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape, order='F')  # Fortran order
  

for _, row in df.iterrows():
    img_name = row['ImageId']
    class_id = int(row['ClassId']) - 1   # make 0-indexed for YOLO
    rle = row['EncodedPixels']

    # Load image to get size
    img_path = os.path.join(images_dir, img_name)
    if not os.path.exists(img_path):
        print(f"Warning: Image {img_name} not found!")
        continue
    w, h = Image.open(img_path).size

    # Decode RLE to mask
    mask = rle_decode(rle, (h, w))  # shape=(height,width)

    # Find contours (polygons)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    label_path = os.path.join(labels_dir, os.path.splitext(img_name)[0] + ".txt")

    with open(label_path, "a") as f:
        for contour in contours:
            if len(contour) < 3:  # skip tiny/noisy contours
                continue
            # Flatten and normalize
            norm_coords = []
            for point in contour:
                x, y = point[0]
                norm_coords.append(x / w)
                norm_coords.append(y / h)

            # Write YOLO seg format
            f.write(f"{class_id} " + " ".join([f"{c:.6f}" for c in norm_coords]) + "\n")


# import pandas as pd

# csv_file = "/home/extra_space/akhilesh/severstal-steel-defect-detection/train.csv"
# df = pd.read_csv(csv_file)

# print(df.head())   # show first 5 rows with headers
# print(df.columns)  # print actual column names
