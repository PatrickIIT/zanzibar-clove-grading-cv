# Zanzibar Clove Grading — Decomposed Multi-Task Vision Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Hugging Face Dataset](https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-zanzibar__cloves-yellow)](https://huggingface.co/datasets/PatrickIITMZ/zanzibar_cloves)
[![CVPR V4A 2026](https://img.shields.io/badge/CVPR%202026-V4A%20Workshop-red)](https://cvpr.thecvf.com)
[![EAC STI 2026](https://img.shields.io/badge/EAC%20STI%202026-Accepted-green)](https://eac.int)
[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-orange.svg)](https://pytorch.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://tensorflow.org)

Official code, models, dataset, and mobile application for the M.Tech thesis:

> **"Decomposed Multi-Task Vision for Auditable Agricultural Grading:  
> A Study on Rule-Based Clove Classification"**  
> Patrick Vincent Ndowo — IIT Madras Zanzibar Campus, June 2026  
> Supervisor: Dr. Innocent Nyalala

---

## The Problem This Solves

Prior computer vision work on clove quality assessment treated grading as a
single-step image classification task: feed one clove image into a CNN, get a
grade label out. This achieves high accuracy on benchmark test sets but is
**structurally incapable** of replicating what the Zanzibar State Trading
Corporation (ZSTC) actually does during grading — an ordered, multi-step
procedure involving batch examination, counting of *mpeta* (fermented cloves),
proportion estimation, and deterministic application of official quantitative
thresholds.

This repository contains a framework that **closes that gap** by mirroring the
ZSTC procedural logic computationally, producing grade decisions that are not
only accurate but independently verifiable by a ZSTC officer without any
specialised AI knowledge.

---

## Results at a Glance

| Mode | Model | Accuracy | Deployment |
|---|---|---|---|
| Single-Clove (High-Precision) | YOLOv8-seg + Context-Aware ResNet-18 | **99.45%** | Android / Offline |
| Batch-Clove (Efficiency) | EfficientNet-Lite0 INT8 TFLite | **84.56%** | Android / Offline |
| Single-Clove Backbone (standalone) | Context-Aware ResNet-18 | **99.02%** | — |
| Best Monolithic Baseline | ResNet-50 / ResNet-101 / DenseNet-201 / EfficientNet-B2 | **99.71%** | — |
| Classical ML Baseline | SVM-RBF (93-dim CIELAB + GLCM features) | **96.23%** | — |

> The 22-architecture benchmark is the most comprehensive evaluation of deep
> learning models for clove grading published to date.

---

## Key Features

- **Dual-path decomposed framework**
  - *Single-clove mode*: YOLOv8-seg generates pixel-level instance masks →
    4-channel RGB+mask input → context-aware ResNet-18 → rule engine computes
    *mpeta* fraction from instance counts → ZSTC threshold applied
    deterministically
  - *Batch-clove mode*: EfficientNet-Lite0 classifies pile photographs directly
    → rule engine maps predicted grade to ZSTC threshold bracket → audit trail
    generated as practical approximation where pile-level instance separation is
    operationally impractical
- **Rule-based grading engine**: applies official ZSTC quantitative thresholds
  (*mpeta* ≤ 3% Grade I, ≤ 7% Grade II, ≤ 20% Grade III, > 20% Grade IV) as
  deterministic program logic — not as a training objective
- **Human-readable audit trail**: every decision records the assigned grade,
  computed or inferred *mpeta* fraction, and the ZSTC threshold applied —
  verifiable without AI expertise
- **INT8 TFLite deployment**: both models < 5 MB, integrated into the
  **Ubora-AI Flutter application** (Swahili: *ubora* = quality), operating
  fully offline on mid-range Android hardware
- **Comprehensive benchmark**: 22 architectures across 6 families — classical
  ML (SVM, Random Forest), CNN Residual (ResNet-18/50/101), CNN Dense
  (DenseNet-121/169/201), Efficient Scaling (EfficientNet-B0–B4), Multi-Scale /
  Hybrid (Inception-v3, Inception-ResNet-v2, Xception), Vision Transformers
  (ViT-B/16, DeiT-B, Swin-B), plus VGG-16 as negative baseline
- **Open dataset**: first publicly released ZSTC-graded Zanzibar clove dataset —
  5,898 images (5,298 single-clove + 600 batch-pile), four official grades,
  labels confirmed by three ZSTC-certified inspectors

---

## Repository Structure

```
zanzibar-clove-grading-cv/
│
├── README.md
├── requirements.txt
├── LICENSE
│
├── notebooks/                         # Reproducible experiment notebooks
│   ├── 01_classical_ml_baseline.ipynb # SVM + Random Forest, CIELAB features,
│   │                                  # GLCM ablation (96.23% F1)
│   ├── 02_resnet_benchmark.ipynb      # ResNet-18/50/101 (94.34–99.71% F1)
│   ├── 03_densenet_benchmark.ipynb    # DenseNet-121/169/201 (99.57–99.71% F1)
│   ├── 04_efficientnet_benchmark.ipynb# EfficientNet-B0–B4 (98.42–99.71% F1)
│   ├── 05_inception_xception.ipynb    # Inception-v3, Inception-ResNet-v2,
│   │                                  # Xception (99.13–99.28% F1)
│   ├── 06_vit_benchmark.ipynb         # ViT-B/16 (95.05% F1)
│   ├── 07_deit_benchmark.ipynb        # DeiT-B (99.57% F1)
│   ├── 08_swin_benchmark.ipynb        # Swin-B (99.57% F1)
│   ├── 09_unet_segmentation.ipynb     # U-Net semantic segmentation baseline
│   │                                  # (Dice=0.9505 single-clove;
│   │                                  #  failed on dense pile images)
│   ├── 10_yolov8_seg.ipynb            # YOLOv8-seg instance segmentation
│   │                                  # (box mAP50=0.980, mask mAP50=0.979)
│   ├── 11_context_aware_resnet18.ipynb# 4-channel RGB+mask classifier
│   │                                  # (99.02% standalone, 99.45% pipeline)
│   ├── 12_efficientnet_lite0_batch.ipynb # Batch-clove classifier with mixup,
│   │                                  # label smoothing, domain-gap aug,
│   │                                  # COCO negatives (84.56% accuracy)
│   └── 13_tflite_export_quantisation.ipynb # INT8/FP16/dynamic export and
│                                      # accuracy verification
│
├── src/                               # Core reusable modules
│   ├── dataset/
│   │   ├── clove_dataset.py           # PyTorch Dataset for single-clove images
│   │   ├── batch_dataset.py           # TensorFlow Dataset for pile images
│   │   ├── mask_dataset.py            # 4-channel RGB+mask dataset loader
│   │   └── coco_processor.py         # CVAT COCO annotation → binary masks
│   ├── models/
│   │   ├── context_aware_resnet18.py  # Modified ResNet-18 (3→4 input channels,
│   │   │                              # mean-initialised 4th channel)
│   │   └── efficientnet_lite0_tf.py   # EfficientNet-Lite0 TF/Keras definition
│   ├── segmentation/
│   │   ├── unet.py                    # Enhanced U-Net (BCE+Dice loss)
│   │   └── watershed_attempt.py      # Distance-transform watershed baseline
│   │                                  # (documented failure on pile images)
│   ├── features/
│   │   └── handcrafted_features.py    # 93-dim CIELAB: colour histograms (64),
│   │                                  # statistical moments (9), GLCM (20)
│   ├── rule_engine/
│   │   └── zstc_grade_engine.py       # Deterministic ZSTC threshold logic;
│   │                                  # audit trail generation
│   └── utils/
│       ├── training.py                # Train / validate loops, early stopping
│       ├── evaluation.py              # Macro F1, confusion matrix, per-class
│       └── visualisation.py          # Benchmark lollipop, training curves,
│                                      # feature ablation, quantisation scatter
│
├── models/                            # Trained model weights
│   ├── yolov8_clove_seg/
│   │   └── best.pt                   # YOLOv8-seg weights (box mAP50=0.980)
│   ├── context_aware_resnet18/
│   │   └── best_context_aware_model.pth  # 4-channel ResNet-18 (99.02%)
│   ├── tflite/
│   │   ├── clove_grader_int8.tflite  # EfficientNet-Lite0 INT8 (4.1 MB)
│   │   ├── clove_grader_fp16.tflite  # EfficientNet-Lite0 FP16 (7.1 MB)
│   │   └── clove_grader_dynamic.tflite # Dynamic-range (3.9 MB)
│   └── classical/
│       ├── svm_rbf_clove.pkl         # SVM-RBF (C=10, γ=scale, 96.23% F1)
│       └── random_forest_clove.pkl   # Random Forest (n=200, 92.11% F1)
│
├── datasets/                          # Dataset access instructions
│   └── README_dataset.md             # How to download from Hugging Face;
│                                      # dataset composition; annotation protocol
│
└── app/                               # Ubora-AI Flutter Android application
    ├── lib/
    │   ├── ml_service.dart           # TFLite inference service
    │   ├── image_processor.dart      # Preprocessing (resize, normalise)
    │   ├── rule_engine.dart          # ZSTC threshold logic + audit trail
    │   └── rag_service.dart          # RAG-based AI Advisor (ZSTC knowledge base)
    └── pubspec.yaml                  # Flutter dependencies (tflite_flutter,
                                       # image, camera)
```

---

## Dataset

The dataset is publicly available on Hugging Face:

```
https://huggingface.co/datasets/PatrickIITMZ/zanzibar_cloves
```

| Split | Grade I | Grade II | Grade III | Grade IV | Total |
|---|---|---|---|---|---|
| Single-clove | 1,084 | 1,050 | 1,503 | 966 | **5,298** |
| Batch-pile | 150 | 150 | 150 | 150 | **600** |
| **Total** | **1,234** | **1,200** | **1,653** | **1,116** | **5,898** |

**Collection details:**
- Camera: Samsung Galaxy S21 Ultra 5G (SM-G998B/DS), 12 MP, f/1.8
- Location: ZSTC Saateni warehouse, Unguja, Zanzibar
- Phase 1 (2024): handheld, natural ambient warehouse light
- Phase 3 (March–May 2026): PULUZ LED lightbox + tripod + fixed overhead mount
- Labels: consensus of three ZSTC-certified inspectors across all images
- Pixel-level annotations: 200 images annotated in Roboflow (polygon masks,
  COCO format) for YOLOv8-seg and context-aware ResNet-18 training

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/PatrickIIT/zanzibar-clove-grading-cv.git
cd zanzibar-clove-grading-cv
pip install -r requirements.txt
```

### 2. Download the dataset

```python
from datasets import load_dataset
ds = load_dataset("PatrickIITMZ/zanzibar_cloves")
```

### 3. Run the rule engine on a single clove image

```python
from ultralytics import YOLO
from src.models.context_aware_resnet18 import ContextAwareResNet18
from src.rule_engine.zstc_grade_engine import ZSTCGradeEngine

# Load models
seg_model  = YOLO("models/yolov8_clove_seg/best.pt")
cls_model  = ContextAwareResNet18.load("models/context_aware_resnet18/best_context_aware_model.pth")
rule_engine = ZSTCGradeEngine()

# Run single-clove pipeline
results      = seg_model("path/to/clove.jpg")
mask         = results[0].masks.data[0]          # pixel-level binary mask
grade_logits = cls_model.predict_with_mask("path/to/clove.jpg", mask)
grade, audit = rule_engine.assign_grade(grade_logits)

print(f"Grade: {grade}")
print(f"Audit trail: {audit}")
# Example output:
# Grade: Grade I
# Audit trail: Grade I assigned. Mpeta = 2.3%. Threshold applied: mpeta <= 3%.
```

### 4. Run the batch-clove classifier

```python
import tensorflow as tf
import numpy as np
from PIL import Image
from src.rule_engine.zstc_grade_engine import ZSTCGradeEngine

# Load INT8 TFLite model
interpreter = tf.lite.Interpreter("models/tflite/clove_grader_int8.tflite")
interpreter.allocate_tensors()

# Preprocess pile image
img = Image.open("path/to/pile.jpg").resize((224, 224))
inp = np.expand_dims(np.array(img) / 255.0, 0).astype(np.float32)

# Inference
interpreter.set_tensor(interpreter.get_input_details()[0]['index'], inp)
interpreter.invoke()
logits     = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
pred_class = np.argmax(logits)

# Rule engine: threshold lookup from predicted label
rule_engine = ZSTCGradeEngine()
grade, audit = rule_engine.lookup_grade(pred_class)
print(f"Grade: {grade}")
print(f"Audit trail: {audit}")
```

### 5. Reproduce the full benchmark

```bash
# Run all 22 architectures sequentially
jupyter nbconvert --to notebook --execute notebooks/02_resnet_benchmark.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_densenet_benchmark.ipynb
# ... (see notebooks/ for all 13 notebooks)
```

---

## ZSTC Official Grading Standard

The rule engine implements these thresholds exactly as codified by ZSTC:

| Grade | Primary Attribute | Mpeta (max) | Foreign Matter (max) | Moisture (max) |
|---|---|---|---|---|
| Grade I | Attractive golden colour | ≤ 3% | ≤ 5% | 14% |
| Grade II | Faded / blackish colour | ≤ 7% | ≤ 5% | 14% |
| Grade III | More faded colour | ≤ 20% | ≤ 5% | Brittleness test |
| Grade IV | Primarily mpeta | > 20% | N/A | N/A |

> *Mpeta* (Swahili) = *khoker* (international trade term): fermented cloves
> from which the flower bud has separated or collapsed.

The price differential between Grade I (TZS 15,000/kg) and Grade IV
(TZS 7,000/kg) exceeds **53%** — making accurate, auditable grading an
economic necessity for Zanzibar farmers.

---

## Experiment Summary

| Experiment | Notebook | Key result |
|---|---|---|
| Classical ML — SVM + Random Forest, CIELAB + GLCM | `01` | SVM 96.23% F1 |
| GLCM feature ablation (7 subsets) | `01` | Colour+Stats+Texture = best |
| ResNet-18 / 50 / 101 benchmark | `02` | 94.34% / 99.71% / 99.71% |
| DenseNet-121 / 169 / 201 benchmark | `03` | 99.57% / 99.57% / 99.71% |
| EfficientNet-B0–B4 benchmark | `04` | B2 = 99.71%; non-monotonic scaling |
| Inception-v3 / Inception-ResNet-v2 / Xception | `05` | 99.13% / 99.13% / 99.28% |
| ViT-B/16 benchmark | `06` | 95.05% (no spatial locality prior) |
| DeiT-B benchmark | `07` | 99.57% (distillation transfers CNN priors) |
| Swin-B benchmark | `08` | 99.57% (hierarchical local windows) |
| U-Net semantic segmentation baseline | `09` | Dice=0.9505 single-clove; failed on piles |
| Watershed on U-Net masks | `09` | Failed: merged blobs on dense overlaps |
| YOLOv8-seg instance segmentation | `10` | box mAP50=0.980, mask mAP50=0.979 |
| Context-aware ResNet-18 (RGB+mask 4-ch) | `11` | 99.02% standalone; 99.45% pipeline |
| EfficientNet-Lite0 batch classifier | `12` | 84.56% (Grade I recall=0.36 main gap) |
| INT8 / FP16 / dynamic TFLite export | `13` | INT8=80.15%, 4.1 MB deployed |

---

## Architecture: Decomposed Multi-Task Vision

```
SINGLE-CLOVE PATH (High-Precision)
────────────────────────────────────────────────────────────────────
Input RGB 224×224×3
    │
    ▼
YOLOv8-seg ──► BBox [x,y,w,h]
    │
    ▼
Mask Head M ∈ ℝ^{H×W×1}
    │
    ▼
4-Channel Concat  224×224×3 ⊕ 224×224×1 = X_joint ∈ ℝ^{224×224×4}
    │
    ▼
Context-Aware ResNet-18
  Conv1 (k=7,s=2, 4-ch)  → 112×112×64
  Res Blocks 1–2          → 56×56×128
  Res Blocks 3–4          → 28×28×256
  Res Blocks 5–6          → 14×14×512
  GAP → FC → Softmax      → ŷ ∈ ℝ^{4×1}
    │
    ▼
Rule Engine
  Accumulate per-instance labels across batch
  γ = count(mpeta) / count(all instances)
  Grade I:   γ ≤ 0.03
  Grade II:  0.03 < γ ≤ 0.07
  Grade III: 0.07 < γ ≤ 0.20
  Grade IV:  γ > 0.20
  → Human-readable audit trail


BATCH-CLOVE PATH (Efficiency)
────────────────────────────────────────────────────────────────────
Input Pile Image X_pile ∈ ℝ^{224×224×3}
    │
    ▼
EfficientNet-Lite0
  MBConv1 Blocks (Swish) → 112×112×16
  MBConv6 Blocks (DW Sep Conv) → 7×7×320
  GAP + Dropout → v_B ∈ ℝ^{1280×1}
  FC → Softmax → ŷ ∈ ℝ^{5×1}  [Grade I–IV + Not_Clove]
    │
    ▼
INT8 Post-Training Quantisation  ℝ → ℤ
  → model.tflite  4.1 MB
    │
    ▼
Flutter Engine (offline)
    │
    ▼
Rule Engine
  Map predicted label → ZSTC threshold bracket
  → Audit trail (grade + applicable threshold stated)
```

---

## Publications from This Work

- **Nyalala, I. & Vincent, P.** (2026). CLOVES-4603: Benchmarking Classical
  Texture Features and Fine-Tuned Deep Models for Clove Quality Grading.
  *CVPR 2026 — Vision for Agriculture (V4A) Workshop* (Poster). [Accepted]

- **Vincent, P. & Nyalala, I.** (2026). Deep Learning for Clove Quality
  Classification: Comparing CNN Architectures for Zanzibar's Export Sector.
  *4th EAC Regional Science, Technology and Innovation Conference, Kigali,
  Rwanda*. [Accepted]

- **Vincent, P. & Nyalala, I.** (2026). Unifying Perspectives on Learning
  Biases: A Data-Centric Intervention for Holistic Fairness, Robustness, and
  Generalization. *ICLR 2026 Workshop on Principled Design for Trustworthy AI*.
  [Accepted]

- **Vincent, P. & Nyalala, I.** (2025). From Art to Algorithms: Co-Designing
  AI for Clove Grading with Zanzibar's Indigenous Experts.
  *AfriCHI 2025 Workshop on Advancing Sustainable Agricultural Practices in
  Africa with AI, Cairo, Egypt*. [Accepted]

---

## Citation

If you use this code, dataset, or findings in your research, please cite:

```bibtex
@mastersthesis{vincent2026clove,
  author    = {Vincent Ndowo, Patrick},
  title     = {Decomposed Multi-Task Vision for Auditable Agricultural Grading:
               A Study on Rule-Based Clove Classification},
  school    = {Indian Institute of Technology Madras, Zanzibar Campus},
  year      = {2026},
  month     = {June},
  address   = {Bweleo, Zanzibar, Tanzania},
  note      = {Supervisor: Dr.\ Innocent Nyalala}
}

@dataset{vincent2026cloves4603,
  author    = {Vincent Ndowo, Patrick and Nyalala, Innocent},
  title     = {CLOVES-4603: Zanzibar Clove Grading Dataset},
  year      = {2026},
  publisher = {Hugging Face},
  url       = {https://huggingface.co/datasets/PatrickIITMZ/zanzibar_cloves}
}
```

---

## Acknowledgements

- **Dr. Innocent Nyalala** — research supervisor, IIT Madras Zanzibar Campus
- **Zanzibar State Trading Corporation (ZSTC)** — field access, grading
  expertise, and official standards documentation (Saateni warehouse, Unguja)
- **ZSTC-certified graders** — annotation and ground-truth label verification
- **IIT Madras Zanzibar Campus** — institutional support and academic environment
- Compute resources: Google Colab Pro and Kaggle Notebooks (NVIDIA T4 GPU)

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for
details.

The dataset is released under
[Creative Commons Attribution 4.0 (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
When using the dataset, please cite the CLOVES-4603 paper above.

---

*Ubora-AI — ubora* is the Swahili word for **quality**.
