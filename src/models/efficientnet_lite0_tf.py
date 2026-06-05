"""
efficientnet_lite0_tf.py
------------------------
EfficientNet-Lite0 classification model for batch-pile (group-clove) grading.

Architecture
~~~~~~~~~~~~
- Backbone: EfficientNet-Lite0 loaded from TensorFlow Hub (frozen).
  Output: 1280-dimensional feature vector.
- Head: Dropout(0.3) → Dense(256, swish) → BatchNorm → Dropout(0.2)
  → Dense(N_CLASSES, softmax).
- Classes: Grade_1, Grade_2, Grade_3, Grade_4, Not_Clove (5 total).
  The Not_Clove class uses COCO negatives as a rejection class.

Training strategy (two-phase)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Phase 1: Backbone frozen — train head only (lr=1e-3, 30 epochs).
Phase 2: Backbone unfreeze skipped (TF1 Hub format limitation);
         frozen backbone used throughout.

Regularisation
~~~~~~~~~~~~~~
- Mixup (α=0.2) — see batch_dataset.py
- Label smoothing (ε=0.1) inside CategoricalCrossentropy
- Domain-gap augmentation — see batch_dataset.py

Deployment
~~~~~~~~~~
Trained model exported to three TFLite variants:
- INT8 post-training quantisation  → 4.1 MB  (80.15% test accuracy)
- FP16 post-training quantisation  → 7.1 MB  (81.62% test accuracy)
- Dynamic-range quantisation       → 3.9 MB  (80.88% test accuracy)
Full-precision test accuracy: 84.56% on 136-image held-out test set.

Reference
---------
Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for
convolutional neural networks. ICML 2019. https://arxiv.org/abs/1905.11946
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMG_SIZE      = 224
N_CLASSES     = 5
CLASS_NAMES   = ["Grade_1", "Grade_2", "Grade_3", "Grade_4", "Not_Clove"]
LABEL_SMOOTH  = 0.1
LR_HEAD       = 1e-3
EPOCHS_HEAD   = 30
BATCH_SIZE    = 32

# TF Hub URL for EfficientNet-Lite0 feature vector (TF1 SavedModel format)
EFFNET_LITE0_URL = (
    "https://tfhub.dev/tensorflow/efficientnet/lite0/feature-vector/2"
)


# --------------------------------------------------------------------------- #
# Model builder
# --------------------------------------------------------------------------- #
def build_model(
    trainable_backbone: bool = False,
    hub_url: str = EFFNET_LITE0_URL,
    n_classes: int = N_CLASSES,
) -> keras.Model:
    """Build the EfficientNet-Lite0 + classification head.

    Parameters
    ----------
    trainable_backbone : bool
        If True, unfreezes the TF Hub backbone for fine-tuning.
        Note: the TF1 SavedModel Hub format used for Lite0 does not
        support unfreeze in TF2; set to False (default) to avoid errors.
    hub_url : str
        TensorFlow Hub URL for the EfficientNet-Lite0 feature extractor.
    n_classes : int
        Number of output classes (default 5: Grade 1–4 + Not_Clove).

    Returns
    -------
    keras.Model
        Uncompiled Keras model.
    """
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")

    backbone = hub.KerasLayer(
        hub_url,
        trainable=trainable_backbone,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        name="efficientnet_lite0",
    )

    x = backbone(inputs)                                      # (batch, 1280)
    x = keras.layers.Dropout(0.3,  name="dropout_1")(x)
    x = keras.layers.Dense(256, activation="swish", name="fc1")(x)
    x = keras.layers.BatchNormalization(name="bn1")(x)
    x = keras.layers.Dropout(0.2, name="dropout_2")(x)
    outputs = keras.layers.Dense(n_classes, activation="softmax",
                                 name="predictions")(x)

    return keras.Model(inputs, outputs, name="clove_grader_lite0")


def compile_model(
    model: keras.Model,
    learning_rate: float = LR_HEAD,
    label_smoothing: float = LABEL_SMOOTH,
) -> keras.Model:
    """Compile the model with Adam + label-smoothed cross-entropy.

    Parameters
    ----------
    model : keras.Model
        Model returned by :func:`build_model`.
    learning_rate : float
        Initial learning rate.
    label_smoothing : float
        Label smoothing coefficient ε (applied inside the loss).

    Returns
    -------
    keras.Model
        Compiled model (in-place, also returned for convenience).
    """
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.CategoricalCrossentropy(
            label_smoothing=label_smoothing
        ),
        metrics=["accuracy"],
    )
    return model


# --------------------------------------------------------------------------- #
# Callbacks
# --------------------------------------------------------------------------- #
def get_callbacks(
    run_name: str,
    output_dir: str | Path = "outputs",
    monitor: str = "val_accuracy",
) -> list:
    """Return standard callbacks for EfficientNet-Lite0 training.

    Includes ModelCheckpoint (best val_accuracy), ReduceLROnPlateau
    (patience=4, factor=0.5), and EarlyStopping (patience=8).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    return [
        keras.callbacks.ModelCheckpoint(
            str(output_dir / f"best_{run_name}.keras"),
            monitor=monitor,
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor=monitor,
            factor=0.5,
            patience=4,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=8,
            restore_best_weights=True,
            verbose=1,
        ),
    ]


# --------------------------------------------------------------------------- #
# TFLite export
# --------------------------------------------------------------------------- #
def export_tflite(
    model: keras.Model,
    output_dir: str | Path,
    representative_dataset_fn=None,
) -> dict[str, Path]:
    """Export the trained model to INT8, FP16, and dynamic TFLite variants.

    Parameters
    ----------
    model : keras.Model
        Trained Keras model (post Phase 1 training, best weights loaded).
    output_dir : str | Path
        Directory where ``.tflite`` files will be saved.
    representative_dataset_fn : callable, optional
        Generator that yields batches of representative input data for
        INT8 calibration (required for full-integer quantisation).

    Returns
    -------
    dict[str, Path]
        Mapping from variant name to output file path::

            {"int8": Path(...), "fp16": Path(...), "dynamic": Path(...)}
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as SavedModel first
    saved_model_dir = output_dir / "saved_model"
    model.save(str(saved_model_dir))

    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    paths: dict[str, Path] = {}

    # ── INT8 ────────────────────────────────────────────────────────
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if representative_dataset_fn:
        converter.representative_dataset = representative_dataset_fn
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type  = tf.uint8
        converter.inference_output_type = tf.uint8
    tflite_int8 = converter.convert()
    p = output_dir / "clove_grader_int8.tflite"
    p.write_bytes(tflite_int8)
    paths["int8"] = p
    print(f"INT8  model saved → {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    # ── FP16 ────────────────────────────────────────────────────────
    converter2 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter2.optimizations = [tf.lite.Optimize.DEFAULT]
    converter2.target_spec.supported_types = [tf.float16]
    tflite_fp16 = converter2.convert()
    p = output_dir / "clove_grader_fp16.tflite"
    p.write_bytes(tflite_fp16)
    paths["fp16"] = p
    print(f"FP16  model saved → {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    # ── Dynamic ─────────────────────────────────────────────────────
    converter3 = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
    converter3.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dyn = converter3.convert()
    p = output_dir / "clove_grader_dynamic.tflite"
    p.write_bytes(tflite_dyn)
    paths["dynamic"] = p
    print(f"Dyn.  model saved → {p}  ({p.stat().st_size / 1e6:.1f} MB)")

    return paths


# --------------------------------------------------------------------------- #
# Single-image inference helper
# --------------------------------------------------------------------------- #
def predict_tflite(
    tflite_path: str | Path,
    image: np.ndarray,
) -> Tuple[int, np.ndarray]:
    """Run inference on a single pre-processed image with a TFLite model.

    Parameters
    ----------
    tflite_path : str | Path
        Path to a ``.tflite`` model file.
    image : np.ndarray
        Pre-processed image of shape ``(224, 224, 3)``, dtype float32,
        values in [0, 1].

    Returns
    -------
    pred_class : int
        Predicted class index (0–4).
    probabilities : np.ndarray
        Softmax probability vector of shape ``(N_CLASSES,)``.
    """
    interpreter = tf.lite.Interpreter(str(tflite_path))
    interpreter.allocate_tensors()
    inp_detail = interpreter.get_input_details()[0]
    out_detail = interpreter.get_output_details()[0]

    batch = np.expand_dims(image.astype(np.float32), axis=0)
    interpreter.set_tensor(inp_detail["index"], batch)
    interpreter.invoke()
    logits = interpreter.get_tensor(out_detail["index"])[0]

    pred_class = int(np.argmax(logits))
    return pred_class, logits
