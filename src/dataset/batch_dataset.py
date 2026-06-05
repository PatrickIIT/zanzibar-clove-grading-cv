"""
batch_dataset.py
----------------
TensorFlow / Keras tf.data pipeline for batch-pile (group-clove) images.

Used in notebook/12 (EfficientNet-Lite0 batch classifier).

Classes
-------
0 → Grade_1
1 → Grade_2
2 → Grade_3
3 → Grade_4
4 → Not_Clove   (COCO negatives — rejection class)

Folder layout expected::

    root/
    ├── Grade_1/
    ├── Grade_2/
    ├── Grade_3/
    ├── Grade_4/
    └── Not_Clove/

Training pipeline includes:
- Random brightness / contrast / saturation (domain-gap augmentation)
- Random 90° rotations
- Mixup (α = 0.2) applied at batch level
- Label smoothing (ε = 0.1) applied inside the loss, not the dataset
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import tensorflow as tf

# --------------------------------------------------------------------------- #
# Constants (must match EfficientNet-Lite0 input spec)
# --------------------------------------------------------------------------- #
IMG_SIZE    = 224
N_CLASSES   = 5
MIXUP_ALPHA = 0.2

CLASS_NAMES: list[str] = ["Grade_1", "Grade_2", "Grade_3", "Grade_4", "Not_Clove"]
CLASS_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(CLASS_NAMES)}


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _decode_image(path: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Read JPEG/PNG from disk, decode, resize, normalise to [0, 1]."""
    raw   = tf.io.read_file(path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE])
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def _augment_train(image: tf.Tensor, label: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Domain-gap augmentation for training batches.

    Replicates the augmentation strategy used in Experiment 4
    (EfficientNet-Lite0 batch classifier) to compensate for
    inter-session chromatic variability between Phase 1 and Phase 3
    data collection.
    """
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # Random 90° rotation (k ∈ {0,1,2,3})
    k     = tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image, label


def _mixup_batch(
    images: tf.Tensor,
    labels: tf.Tensor,
    alpha: float = MIXUP_ALPHA,
) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply mixup augmentation within a batch.

    Interpolates between randomly paired (image, label) pairs using a
    Beta(α, α)-distributed mixing coefficient λ.  Particularly helpful
    at the Grade I / Grade II boundary where visual separation is narrow.

    Reference: Zhang et al. (2018) "mixup: Beyond Empirical Risk Minimization."
    https://arxiv.org/abs/1710.09412
    """
    if alpha == 0:
        return images, labels
    batch_size = tf.shape(images)[0]
    lam = np.random.beta(alpha, alpha, size=(int(images.shape[0] or 32), 1, 1, 1))
    lam = tf.constant(lam, dtype=tf.float32)

    indices  = tf.random.shuffle(tf.range(batch_size))
    images2  = tf.gather(images, indices)
    labels2  = tf.gather(labels, indices)

    mixed_images = lam * images + (1.0 - lam) * images2
    mixed_labels = tf.reshape(lam, [-1, 1]) * labels + \
                   (1.0 - tf.reshape(lam, [-1, 1])) * labels2
    return mixed_images, mixed_labels


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_file_list(
    root: str | Path,
    classes: Optional[List[str]] = None,
) -> Tuple[List[str], List[int]]:
    """Walk *root* and collect (path, label) pairs for all grade folders.

    Parameters
    ----------
    root :
        Dataset root directory containing one sub-folder per class.
    classes :
        Ordered list of sub-folder names.  Defaults to
        ``CLASS_NAMES`` (Grade_1 … Not_Clove).

    Returns
    -------
    paths : list[str]
    labels : list[int]
    """
    root    = Path(root)
    classes = classes or CLASS_NAMES
    paths, labels = [], []
    extensions = {".jpg", ".jpeg", ".png", ".bmp"}

    for cls_name in classes:
        label  = CLASS_TO_IDX[cls_name]
        folder = root / cls_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Class folder not found: {folder}")
        for f in sorted(folder.iterdir()):
            if f.suffix.lower() in extensions:
                paths.append(str(f))
                labels.append(label)

    return paths, labels


def make_dataset(
    paths: List[str],
    labels: List[int],
    training: bool = False,
    batch_size: int = 32,
) -> tf.data.Dataset:
    """Build a prefetched tf.data.Dataset from file-path / label lists.

    Parameters
    ----------
    paths : list[str]
        Absolute paths to image files.
    labels : list[int]
        Integer class labels (0–4).
    training : bool
        If True, applies augmentation and mixup.
    batch_size : int
        Mini-batch size.

    Returns
    -------
    tf.data.Dataset yielding (image_batch, label_batch) tensors.
    Labels are one-hot encoded (shape: [batch_size, N_CLASSES]).
    """
    labels_oh = tf.one_hot(labels, N_CLASSES)
    ds = tf.data.Dataset.from_tensor_slices(
        (tf.constant(paths), labels_oh)
    )
    if training:
        ds = ds.shuffle(len(paths), reshuffle_each_iteration=True)

    ds = ds.map(_decode_image, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.map(_augment_train, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=training)

    if training and MIXUP_ALPHA > 0:
        ds = ds.map(
            lambda x, y: tf.numpy_function(
                lambda imgs, lbls: _mixup_batch(imgs, lbls, MIXUP_ALPHA),
                [x, y],
                [tf.float32, tf.float32],
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            lambda x, y: (
                tf.ensure_shape(x, [None, IMG_SIZE, IMG_SIZE, 3]),
                tf.ensure_shape(y, [None, N_CLASSES]),
            )
        )

    return ds.prefetch(tf.data.AUTOTUNE)
