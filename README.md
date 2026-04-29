# Zanzibar Clove Grading with Computer Vision

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/PatrickIITMZ/zanzibar_cloves)

This repository contains the official code, models, and mobile app for the Master's thesis:  
**"Decomposed Multi-Task Vision for Auditable Agricultural Grading: Rule-Based Clove Classification in Zanzibar"**  
by Patrick Vincent, IIT Madras Zanzibar Campus.

The work introduces a **procedural, rule-based framework** that mirrors the official ZSTC grading protocol, enabling **auditable, explainable, and accurate** clove quality assessment—with **real-world deployment on Android**.

---

## 📌 Key Features

- ✅ **Dual-path vision system**:
  - **Single-clove mode**: YOLOv8-seg instance segmentation + context-aware ResNet-18 (99.45% accuracy)
  - **Batch-clove mode**: EfficientNet-Lite0 classifier with domain-gap augmentation (84.56% accuracy)
- ✅ **Mobile-ready deployment**: Both models exported to **INT8 TensorFlow Lite** (<5 MB) and integrated into a **Flutter Android app**
- ✅ **Rule-based grading engine**: Applies official ZSTC thresholds (e.g., mpeta ≤ 7% for Grade II) and generates **human-readable audit trails**
- ✅ **Comprehensive benchmark**: 22+ architectures evaluated (ResNet, DenseNet, EfficientNet, ViT, DeiT, Swin, Xception, classical ML)
- ✅ **Open dataset**: First-ever ZSTC-graded clove dataset (5,898 images across 4 grades)

---

## 🗂️ Repository Structure

- `src/` – Core reusable modules (dataset, models, rule engine)
- `notebooks/` – Reproducible experiment notebooks (U-Net, YOLOv8, EfficientNet-Lite0)
- `models/` – Pretrained weights:
  - `best_int8.tflite` – YOLOv8-seg for single-clove mode (3.3 MB)
  - `clove_grader_int8.tflite` – EfficientNet-Lite0 for batch mode (4.1 MB)
- `datasets/` – Instructions to access the full dataset on Hugging Face
- `app/` – Flutter mobile app source code (`ml_service.dart`, `image_processor.dart`)

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
