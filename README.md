#  Rock Paper Scissors Object Detection Project

## 📌 Project Overview

In this project, a **deep learning based object detection model** was developed to recognize three hand gesture classes:

*  Rock
*  Paper
*  Scissors

The goal of this project was to train an **object detection model** capable of detecting and classifying hand gestures in real-time using a webcam .

---

# 1.  Dataset Collection

A custom dataset was collected containing images from three classes:

*  Rock
*  Paper
*  Scissors

Initial dataset information:

* 🖼️ Total images: **523 images**
* 🏷️ Number of classes: **3**

Each image was manually labeled using the **YOLO annotation format**.

YOLO label format:

```text
class_id x_center y_center width height
```

Where:

*  `class_id` represents the object category
*  `x_center` and `y_center` represent normalized bounding box center coordinates
*  `width` and `height` represent normalized bounding box dimensions

---

# 2. 🔄 Data Augmentation

Because the original dataset was limited, **data augmentation** was applied to increase the diversity of training images.

For each original image, **5 augmented versions** were generated.

The augmentation pipeline was implemented using the **Albumentations** library.

##  Geometric Transformations

###  Horizontal Flip

Randomly flips images horizontally:

```text
p = 0.5
```

---

### 📐 Affine Transformation

Includes:

* 🔍 Scaling:

```text
0.85 - 1.15
```

* ↔️ Translation:

```text
-10% to +10%
```

* 🔄 Rotation:

```text
-15 to +15 degrees
```

* 📏 Shearing:

```text
-5 to +5 degrees
```

Probability:

```text
p = 0.8
```

---

## 🎨 Image Appearance Transformations

### ☀️ Random Brightness and Contrast

Simulates different lighting conditions:

```text
brightness_limit = 0.25
contrast_limit = 0.25
```

Probability:

```text
p = 0.7
```

---

### 🌈 Hue Saturation Value Adjustment

Changes image colors:

```text
hue_shift_limit = 10
sat_shift_limit = 20
val_shift_limit = 20
```

Probability:

```text
p = 0.4
```

---

## 🌫️ Noise and Blur

### 🔍 Gaussian Blur

Simulates camera focus changes:

```text
blur_limit = 3-10
p = 0.15
```

---

### 📡 Gaussian Noise

Adds random noise to improve model robustness:

```text
std_range = 0.01-0.04
p = 0.2
```

---

📌 Bounding boxes were also transformed during augmentation to keep YOLO labels synchronized with generated images.

---

# 3. 🧠 Model Training

The augmented dataset was used to train a **YOLO object detection model**.

Training configuration:

* ⏳ Number of epochs:

```text
100 epochs
```

* 🎯 Task:

```text
Object Detection
```

* 🏷️ Classes:

```text
3 classes
```

During training, the model learned:

* 📍 Object localization
* 🏷️ Class prediction
* 📦 Bounding box regression

---

# 4. Real-Time Webcam Detection

After training, the model was connected to a webcam.

The detection pipeline:

1.  Capture live video frames
2.  Send frames to YOLO model
3.  Detect objects
4.  Predict classes:

   *  Rock
   *  Paper
   *  Scissors
5. 📊 Display bounding boxes and confidence scores in real-time

---

# 5. 🔥 Project Pipeline

```text
📸 Image Collection
        |
        ↓
🏷️ Manual YOLO Annotation
        |
        ↓
🔄 Data Augmentation
        |
        ↓
🧠 YOLO Model Training
        |
        ↓
📊 Model Evaluation
        |
        ↓
📷 Webcam Real-Time Detection
```

---

# 6. 🛠️ Technologies Used

* 🐍 Python
* 👁️ OpenCV
* 🔄 Albumentations
* 🧠 YOLO Object Detection
* 🔢 NumPy
* 📷 Webcam Processing

---

# 7. ✅ Results

The final trained model can detect:

*  Rock
*  Paper
*  Scissors

from webcam input in real-time.

The augmentation strategy improved model performance and robustness against:

* 💡 Different lighting conditions
* 📍 Different object positions
* 🔄 Rotation changes
* 🌫️ Image noise
* 📷 Camera variations

---

# 🇮🇷 توضیح فارسی

این پروژه یک سیستم تشخیص حرکت دست با استفاده از **YOLO و Deep Learning** است. مدل با تصاویر سنگ، کاغذ و قیچی آموزش داده شده و می‌تواند از طریق وب‌کم در لحظه آن‌ها را تشخیص دهد.

برای بهتر شدن دقت مدل، از **Data Augmentation** برای افزایش تنوع تصاویر استفاده شده است. این پروژه ترکیبی از **پردازش تصویر، یادگیری عمیق و تشخیص اشیا** است. 
