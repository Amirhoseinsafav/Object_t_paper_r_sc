from pathlib import Path
import cv2
import albumentations as A


# مسیر پوشه دیتاست
DATASET_DIR = Path("C:/Users/Home/Desktop/AXZ/Code/Class project/py/ML/Ml_advance/Cnn/datasets")

IMAGE_DIR = DATASET_DIR / "images" / "train"
LABEL_DIR = DATASET_DIR / "labels" / "train"

# از هر تصویر چند نسخه جدید ساخته شود
AUGMENTATIONS_PER_IMAGE = 5


transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),

        A.Affine(
            scale=(0.85, 1.15),
            translate_percent=(-0.10, 0.10),
            rotate=(-15, 15),
            shear=(-5, 5),
            p=0.8
        ),

        A.RandomBrightnessContrast(
            brightness_limit=0.25,
            contrast_limit=0.25,
            p=0.7
        ),

        A.HueSaturationValue(
            hue_shift_limit=10,
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.4
        ),

        A.GaussianBlur(
            blur_limit=(3, 10),
            p=0.15
        ),

        A.GaussNoise(
            std_range=(0.01, 0.04),
            p=0.2
        ),
    ],

    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.4,
        clip=True
    )
)


def read_yolo_label(label_path):
    bboxes = []
    class_labels = []

    if not label_path.exists():
        return bboxes, class_labels

    with open(label_path, "r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if len(parts) != 5:
                print(f"لیبل نامعتبر: {label_path} -> {line.strip()}")
                continue

            class_id = int(parts[0])
            x_center = float(parts[1])
            y_center = float(parts[2])
            width = float(parts[3])
            height = float(parts[4])

            class_labels.append(class_id)
            bboxes.append([
                x_center,
                y_center,
                width,
                height
            ])

    return bboxes, class_labels


def save_yolo_label(label_path, bboxes, class_labels):
    with open(label_path, "w", encoding="utf-8") as file:
        for class_id, bbox in zip(class_labels, bboxes):
            x_center, y_center, width, height = bbox

            file.write(
                f"{int(class_id)} "
                f"{x_center:.6f} "
                f"{y_center:.6f} "
                f"{width:.6f} "
                f"{height:.6f}/n"
            )


image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

image_paths = [
    path for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in image_extensions
    and "_aug_" not in path.stem
]

print(f"تعداد تصاویر اصلی: {len(image_paths)}")


for image_path in image_paths:
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    if not label_path.exists():
        print(f"لیبل پیدا نشد: {image_path.name}")
        continue

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"تصویر خوانده نشد: {image_path}")
        continue

    bboxes, class_labels = read_yolo_label(label_path)

    if not bboxes:
        print(f"لیبل معتبر ندارد: {label_path.name}")
        continue

    created = 0
    attempts = 0

    while created < AUGMENTATIONS_PER_IMAGE and attempts < 30:
        attempts += 1

        augmented = transform(
            image=image,
            bboxes=bboxes,
            class_labels=class_labels
        )

        augmented_image = augmented["image"]
        augmented_boxes = augmented["bboxes"]
        augmented_classes = augmented["class_labels"]

        # اگر در اثر augmentation همه باکس‌ها حذف شدند، ذخیره نکن
        if len(augmented_boxes) == 0:
            continue

        new_name = f"{image_path.stem}_aug_{created + 1}"

        new_image_path = IMAGE_DIR / f"{new_name}{image_path.suffix.lower()}"
        new_label_path = LABEL_DIR / f"{new_name}.txt"

        saved = cv2.imwrite(
            str(new_image_path),
            augmented_image
        )

        if not saved:
            print(f"ذخیره تصویر ناموفق بود: {new_image_path}")
            continue

        save_yolo_label(
            new_label_path,
            augmented_boxes,
            augmented_classes
        )

        created += 1

    print(
        f"{image_path.name}: "
        f"{created} تصویر جدید ساخته شد"
    )


print("Augmentation تمام شد.")