# Diabetic Retinopathy Detection

This project uses deep learning to classify the severity of diabetic retinopathy from retinal images. It utilizes a convolutional neural network (CNN) model for accurate and efficient classification.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Preprocessing](#preprocessing)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Usage](#usage)

---

## Overview

Diabetic retinopathy is a diabetes complication that affects eyes and can lead to blindness if untreated. This project trains a model to classify retinal images by severity levels, aiding in early detection and management.

## Dataset

Download a dataset such as:
- [APTOS 2019 Blindness Detection](https://www.kaggle.com/c/aptos2019-blindness-detection)
- [EyePACS dataset](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)

Place images in the `data/train` and `data/test` folders, and the labels file in `data/labels.csv`.

## Requirements

Install required packages:

```bash
pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn
```
## Project Structure
.
├── data/

│   ├── train/                  # Training images

│   ├── test/                   # Test images

│   └── labels.csv              # CSV file with image labels

├── notebooks/

│   └── diabetic_retinopathy.ipynb  # Notebook for training and evaluation

├── models/

│   └── best_model.h5           # Saved model weights

└── README.md

## Preprocessing
We perform several preprocessing steps to enhance model accuracy:
Resizing: Images resized to (224, 224).
Normalization: Pixel values scaled to [0, 1] range.
Data Augmentation: Augmentations like rotation, zoom, and horizontal flip are applied to reduce overfitting.
```bash
import cv2
import numpy as np

def preprocess_image(img_path, target_size=(224, 224)):
    img = cv2.imread(img_path)
    img = cv2.resize(img, target_size)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0  # Normalize to [0,1]
    return img
```
## Model Architecture
We use a pretrained ResNet50 model with custom classification layers for transfer learning:
```bash
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Model

# Load base model with pretrained weights
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification layers
x = Flatten()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dense(5, activation='softmax')(x)  # 5 classes for severity

model = Model(inputs=base_model.input, outputs=x)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```
## Training
rain the model using the prepared dataset. Adjust the batch size and epochs as needed.
```bash
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
```
## Evaluation 
Evaluate the model performance on test data
```bash
from sklearn.metrics import classification_report

# Prepare test images and labels
X_test = np.array([preprocess_image(img) for img in test_image_paths])
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)

print(classification_report(y_test, y_pred_classes))
```
## Results
Accuracy: The model achieves an accuracy of 70% on the test set.
## Usage
1.Clone the repository:
```bash
git clone https://github.com/yourusername/diabetic-retinopathy-detection.git
```
2.Install Dependencies:
```bash
pip install -r requirements.txt
```
3.Run the notebook diabetic_retinopathy.ipynb for training and evaluation.

4.Use the saved model (models/best_model.h5) for predictions on new images.

