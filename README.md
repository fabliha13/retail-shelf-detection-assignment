# Intelligent Machines Assignment – Shelf Product Detection & Share-of-Shelf Analysis

**Author:** Fabliha Afaf  
**Date:** March 2026

## Problem Statement

The current object detection model achieves only **67.6% recall**, meaning ~32% of products on shelves are missed. This leads to inaccurate stock counts in retail stores.

**Goals:**
1. Improve **recall** without strongly hurting precision
2. Calculate **Percentage Share of Shelf** for each product class (SKU), treating the entire test set as one representative shelf
3. Deliver clean, reproducible code + results

## Dataset Overview

- Format: YOLO (images + .txt labels with bounding boxes)
- **76 product classes** (SKUs: q1, q4, q7, …, q299)
- Splits:  
  - Train: 924 images  
  - Valid: 40 images  
  - Test: 35 images
- **Strong class imbalance** — a few classes dominate (e.g., q13, q280, q145, q64, q91), many have <20 examples

## Approach

- **Model:** YOLOv8s (small version – good speed/accuracy trade-off)  
- Starting weights: pretrained on COCO (`yolov8s.pt`)  
- Augmentations: mosaic=1.0, mixup=0.15, copy-paste=0.10, HSV, flips, etc. (to help with imbalance)  
- Training: 50 epochs planned, but **early stopping** after 42 epochs (patience=20)  
- Best model: automatically selected from **epoch ~22** (highest validation mAP@0.5)  
- Recall improvement: lower confidence threshold (**0.18** on test) to recover more detections

## Final Results

| Split / Setting              | Precision | Recall   | mAP@0.5 | mAP@0.5:0.95 |
|------------------------------|-----------|----------|---------|--------------|
| Original (given)             | -         | **67.6%** | -       | -            |
| Validation (best weights)    | 76.1%     | 62.5%    | 75.7%   | 44.3%        |
| Validation @ conf=0.15       | 63.3%     | 63.3%    | 64.1%   | 38.2%        |
| **Test @ conf=0.18**         | **72.6%** | **74.6%**| **77.5%**| **43.2%**    |

→ **Recall improved from 67.6% → 74.6%** on the test set  
→ Trade-off: slight precision drop, but clear recall gain (important for missing fewer products)

## Share of Shelf (Test set treated as one shelf)

Calculated by summing detected bounding box widths per class across all test images.

**Top 10 SKUs by percentage:**

- q214 : 13.46%  
- q64  : 13.07%  
- q280 : 10.06%  
- q61  : 4.90%  
- q293 : 4.55%  
- q193 : 4.33%  
- q7   : 3.89%  
- q148 : 3.30%  
- q121 : 3.27%  
- q268 : 3.23%  

(Full list and bar chart in the notebook)

## How to Reproduce

1. Open the notebook in Kaggle or Colab  
2. Install dependencies:  
   ```bash
   !pip install -q ultralytics
