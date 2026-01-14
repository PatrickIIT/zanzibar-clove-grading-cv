# Zanzibar Clove Grading with Computer Vision

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-Hugging%20Face-yellow)](https://huggingface.co/datasets/PatrickIIT/zanzibar_clove)

This repository contains the official code and models for the Master's thesis:  
**"Decomposed Multi-Task Vision for Auditable Agricultural Grading: Rule-Based Clove Classification in Zanzibar"**  
by Patrick Vincent, IIT Madras Zanzibar Campus.

The work introduces a **procedural, rule-based framework** that mirrors the official ZSTC grading protocol, enabling **auditable, explainable, and accurate** clove quality assessment.

---

## 📌 Key Features

- ✅ **Decomposed pipeline**: Instance segmentation → fine-grained classification → rule-based aggregation
- ✅ **Audit trail generation**: Human-readable justification for every grade decision
- ✅ **Benchmark of 22+ models**: ResNet, DenseNet, EfficientNet, ViT, DeiT, Swin, Xception, classical ML
- ✅ **Reproducible results**: Full training scripts, hyperparameters, and evaluation metrics
- ✅ **Open dataset**: First-ever ZSTC-graded clove dataset (5,898 images across 4 grades)

---

## 🗂️ Repository Structure

- `src/` – Core reusable modules (dataset, models, rule engine)
- `notebooks/` – Reproducible experiment notebooks
- `models/` – Pretrained weights (U-Net, Context-Aware Classifier)
- `datasets/` – Instructions to access the full dataset on Hugging Face

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
