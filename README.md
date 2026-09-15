# 🍎🍌🍊 Fruit Image Segmentation using YOLO

A computer vision project that uses **YOLO instance segmentation** to detect,
classify, count, and segment fruits in an image.

## 📌 Project Overview

This project demonstrates how YOLO can be used for **object detection and
instance segmentation**.

The system takes a fruit image as input and produces:

- Individual fruit segmentation masks
- Bounding boxes
- Fruit class labels
- Confidence scores
- Fruit counts

## 🛠️ Technologies Used

- Python
- OpenCV
- NumPy
- Ultralytics YOLO
- YOLO Instance Segmentation

## 📂 Project Structure

```text
fruit-segmentation-classification/
│
├── data/
│   ├── input/
│   │   └── fruits.jpg
│   │
│   └── output/
│       └── yolo/
│           └── fruit_segmentation.jpg
│
├── src/
│   └── yolo_segmentation.py
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
