# Rock Paper Scissors Object Detection Project

## Project Overview

In this project, a deep learning based object detection model was developed to recognize three hand gesture classes:

- Rock
- Paper
- Scissors

The goal of the project was to train an object detection model that can detect and classify these objects in real-time using a webcam.

---

# 1. Dataset Collection

A custom dataset was collected containing images from three classes:

- Rock
- Paper
- Scissors

The initial dataset consisted of:

- Total images: **523 images**
- Number of classes: **3**

Each image was manually labeled using the YOLO annotation format.

YOLO label format:

```
class_id x_center y_center width height
```

where:

- `class_id` represents the object category
- `x_center` and `y_center` represent the normalized center coordinates of the bounding box
- `width` and `height` represent the normalized size of the bounding box

---

# 2. Data Augmentation

Because the original dataset size was limited, data augmentation was applied to increase the diversity of training samples.

For each original image, **5 augmented versions** were generated.

The augmentation pipeline was implemented using the **Albumentations** library.

Applied augmentations:

## Geometric Transformations

### Horizontal Flip

Randomly flips images horizontally with probability:

```
p = 0.5
```

### Affine Transformation

Includes:

- Scaling:
```
0.85 - 1.15
```

- Translation:
```
-10% to +10%
```

- Rotation:
```
-15 to +15 degrees
```

- Shearing:
```
-5 to +5 degrees
```

Probability:

```
p = 0.8
```

---

## Image Appearance Transformations

### Random Brightness and Contrast

Changes illumination conditions:

```
brightness_limit = 0.25
contrast_limit = 0.25
```

Probability:

```
p = 0.7
```

---

### Hue Saturation Value Adjustment

Changes color properties:

```
hue_shift_limit = 10
sat_shift_limit = 20
val_shift_limit = 20
```

Probability:

```
p = 0.4
```

---

## Noise and Blur

### Gaussian Blur

Simulates camera focus changes.

```
blur_limit = 3-10
p = 0.15
```

---

### Gaussian Noise

Adds random noise to images to improve robustness.

```
std_range = 0.01-0.04
p = 0.2
```

---

Bounding boxes were also transformed during augmentation to keep YOLO annotations synchronized with the generated images.

---

# 3. Model Training

The augmented dataset was used to train a YOLO object detection model.

Training configuration:

- Number of epochs:

```
100 epochs
```

- Task:

```
Object Detection
```

- Classes:

```
3 classes
```

During training, the model learned:

- Object localization
- Class prediction
- Bounding box regression

---

# 4. Real-Time Webcam Detection

After training, the trained model was connected to a webcam.

The webcam pipeline performs:

1. Capture live video frames
2. Pass each frame to the YOLO model
3. Detect objects
4. Predict class labels:
   - Rock
   - Paper
   - Scissors
5. Display bounding boxes and confidence scores in real-time

---

# 5. Project Pipeline

The complete workflow:

```
Image Collection
        |
        ↓
Manual YOLO Annotation
        |
        ↓
Data Augmentation
        |
        ↓
YOLO Model Training
        |
        ↓
Model Evaluation
        |
        ↓
Webcam Real-Time Detection
```

---

# 6. Technologies Used

- Python
- OpenCV
- Albumentations
- YOLO Object Detection
- NumPy
- Webcam Processing

---

# 7. Results

The final trained model is capable of detecting Rock, Paper, and Scissors objects from webcam input in real-time.

The augmentation strategy improved the dataset diversity and helped the model become more robust against:

- Different lighting conditions
- Different object positions
- Rotation changes
- Image noise
- Camera variations
