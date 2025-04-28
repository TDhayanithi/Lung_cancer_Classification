import os
import shutil
import random

def split_dataset(source_dir, train_dir, val_dir, split_ratio=0.8):
    classes = os.listdir(source_dir)
    for cls in classes:
        cls_path = os.path.join(source_dir, cls)
        if not os.path.isdir(cls_path):
            continue  # Skip if not a folder

        images = os.listdir(cls_path)
        random.shuffle(images)

        split_index = int(len(images) * split_ratio)
        train_images = images[:split_index]
        val_images = images[split_index:]

        # Create class folders in train and val dirs
        os.makedirs(os.path.join(train_dir, cls), exist_ok=True)
        os.makedirs(os.path.join(val_dir, cls), exist_ok=True)

        for img in train_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(train_dir, cls, img)
            shutil.copy2(src, dst)

        for img in val_images:
            src = os.path.join(cls_path, img)
            dst = os.path.join(val_dir, cls, img)
            shutil.copy2(src, dst)

    print("✅ Dataset successfully split into 'data/train/' and 'data/val/' folders.")

if __name__ == "__main__":
    source_data_dir = "LIDC_Y-Net"   # Update here
    train_output_dir = "data/train"
    val_output_dir = "data/val"

    split_dataset(source_data_dir, train_output_dir, val_output_dir)
