"""
evaluation.py
-------------
Evaluation utilities: macro-averaged F1, confusion matrix, per-class report,
and held-out test set evaluation for both PyTorch and TFLite models.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

GRADE_NAMES = ["Grade I", "Grade II", "Grade III", "Grade IV"]


# --------------------------------------------------------------------------- #
# Core metrics
# --------------------------------------------------------------------------- #
def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Macro-averaged F1-score over all classes.

    Macro-averaging assigns equal weight to each ZSTC grade, regardless
    of its frequency in the dataset.  This is the primary evaluation metric
    used throughout the thesis.
    """
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def per_class_report(
    y_true: List[int],
    y_pred: List[int],
    class_names: Optional[List[str]] = None,
) -> str:
    """Return a formatted per-class precision / recall / F1 report."""
    names = class_names or GRADE_NAMES
    return classification_report(
        y_true, y_pred,
        target_names=names,
        digits=4,
        zero_division=0,
    )


def compute_confusion_matrix(
    y_true: List[int],
    y_pred: List[int],
) -> np.ndarray:
    """Return the confusion matrix as a NumPy array."""
    return confusion_matrix(y_true, y_pred)


# --------------------------------------------------------------------------- #
# PyTorch model evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate_pytorch(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, float, np.ndarray, str]:
    """Evaluate a PyTorch model on a DataLoader.

    Parameters
    ----------
    model : nn.Module
    loader : DataLoader
    device : torch.device, optional
    class_names : list[str], optional

    Returns
    -------
    accuracy : float
    macro_f1_score : float
    conf_matrix : np.ndarray
    report : str
        Full per-class classification report string.
    """
    device = device or (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.eval()
    model.to(device)

    all_preds, all_targets = [], []

    for inputs, targets in loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        preds   = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())

    accuracy  = float(np.mean(np.array(all_preds) == np.array(all_targets)))
    f1        = macro_f1(all_targets, all_preds)
    cm        = compute_confusion_matrix(all_targets, all_preds)
    report    = per_class_report(all_targets, all_preds, class_names)

    print(f"Accuracy:   {accuracy * 100:.2f}%")
    print(f"Macro F1:   {f1 * 100:.2f}%")
    print(f"\nPer-class report:\n{report}")
    return accuracy, f1, cm, report


# --------------------------------------------------------------------------- #
# TFLite model evaluation
# --------------------------------------------------------------------------- #
def evaluate_tflite(
    tflite_path: str | Path,
    image_paths: List[str],
    labels: List[int],
    img_size: int = 224,
    class_names: Optional[List[str]] = None,
) -> Tuple[float, float, np.ndarray, str]:
    """Evaluate a TFLite model on a list of image paths.

    Parameters
    ----------
    tflite_path : str | Path
        Path to the ``.tflite`` model file.
    image_paths : list[str]
        Absolute paths to test images.
    labels : list[int]
        Ground-truth integer labels.
    img_size : int
        Input image size expected by the model (default 224).
    class_names : list[str], optional

    Returns
    -------
    accuracy, macro_f1_score, conf_matrix, report
    """
    import tensorflow as tf
    from PIL import Image

    interpreter = tf.lite.Interpreter(str(tflite_path))
    interpreter.allocate_tensors()
    inp_det = interpreter.get_input_details()[0]
    out_det = interpreter.get_output_details()[0]

    all_preds = []
    for img_path in image_paths:
        img = Image.open(img_path).convert("RGB").resize((img_size, img_size))
        arr = np.array(img, dtype=np.float32) / 255.0
        inp = np.expand_dims(arr, axis=0)
        interpreter.set_tensor(inp_det["index"], inp)
        interpreter.invoke()
        logits = interpreter.get_tensor(out_det["index"])[0]
        all_preds.append(int(np.argmax(logits)))

    accuracy = float(np.mean(np.array(all_preds) == np.array(labels)))
    f1       = macro_f1(labels, all_preds)
    cm       = compute_confusion_matrix(labels, all_preds)
    report   = per_class_report(labels, all_preds, class_names)

    print(f"TFLite model: {Path(tflite_path).name}")
    print(f"Accuracy:   {accuracy * 100:.2f}%")
    print(f"Macro F1:   {f1 * 100:.2f}%")
    print(f"\nPer-class report:\n{report}")
    return accuracy, f1, cm, report
