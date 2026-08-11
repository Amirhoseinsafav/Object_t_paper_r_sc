from pathlib import Path
import random
import shutil

# مسیر اصلی دیتاست
DATASET_DIR = Path("datasets")

# مسیر عکس‌ها و لیبل‌ها
images_train_dir = DATASET_DIR / "images" / "train"
images_val_dir = DATASET_DIR / "images" / "val"

labels_train_dir = DATASET_DIR / "labels" / "train"
labels_val_dir = DATASET_DIR / "labels" / "val"

# ساخت پوشه‌های validation اگر وجود نداشتند
images_val_dir.mkdir(parents=True, exist_ok=True)
labels_val_dir.mkdir(parents=True, exist_ok=True)

# فرمت‌های مجاز عکس
image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

# گرفتن همه عکس‌های train
all_images = [
    img for img in images_train_dir.iterdir()
    if img.suffix.lower() in image_extensions
]

# فقط عکس‌هایی که فایل label هم دارند
valid_images = []
missing_labels = []

for img in all_images:
    label_file = labels_train_dir / f"{img.stem}.txt"

    if label_file.exists():
        valid_images.append(img)
    else:
        missing_labels.append(img.name)

# رندم کردن
random.seed(42)
random.shuffle(valid_images)

# انتخاب 20 درصد برای validation
val_count = int(len(valid_images) * 0.2)
val_images = valid_images[:val_count]

# انتقال عکس و label متناظر
for img in val_images:
    label_file = labels_train_dir / f"{img.stem}.txt"

    shutil.move(str(img), str(images_val_dir / img.name))
    shutil.move(str(label_file), str(labels_val_dir / label_file.name))

print("Done.")
print(f"Total images in train before split: {len(all_images)}")
print(f"Images with labels: {len(valid_images)}")
print(f"Moved to validation: {len(val_images)}")

if missing_labels:
    print("\nThese images did not have matching label files:")
    for name in missing_labels:
        print(name)