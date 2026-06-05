"""
handcrafted_features.py
-----------------------
93-dimensional CIELAB colour + GLCM texture feature extractor.

Feature vector composition
~~~~~~~~~~~~~~~~~~~~~~~~~~~
┌─────────────────────────────────────────┬────────┐
│ Descriptor group                        │  Dims  │
├─────────────────────────────────────────┼────────┤
│ L*a*b* colour histograms                │   64   │
│   a* channel: 32-bin normalised hist    │  (32)  │
│   b* channel: 32-bin normalised hist    │  (32)  │
├─────────────────────────────────────────┼────────┤
│ Statistical colour moments              │    9   │
│   Mean, std, skew for L*, a*, b*        │        │
├─────────────────────────────────────────┼────────┤
│ GLCM texture (L* channel)               │   20   │
│   distance=1, angles={0°,45°,90°,135°} │        │
│   properties: contrast, dissimilarity,  │        │
│               homogeneity, energy,      │        │
│               correlation (×4 angles)  │        │
└─────────────────────────────────────────┴────────┘
Total: 64 + 9 + 20 = 93 dimensions.

CIELAB is used as the working colour space because it provides perceptual
uniformity — equal Euclidean distances correspond to equal perceived colour
differences — which aligns directly with the visual criteria ZSTC graders
apply when distinguishing the golden (Grade I), faded (Grade II), and
oxidised-brown (Grades III–IV) surface colours of cloves.

Best SVM result: 96.23% macro F1 (C=10, γ='scale', RBF kernel).
Best RF result:  92.11% macro F1 (n_estimators=200).

Reference
---------
Haralick, R. M., Shanmugam, K., & Dinstein, I. (1973). Textural features
for image classification. IEEE Transactions on Systems, Man, and
Cybernetics, 3(6), 610–621.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image
from skimage import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops


# --------------------------------------------------------------------------- #
# Individual descriptor functions
# --------------------------------------------------------------------------- #
def extract_colour_histograms(lab_img: np.ndarray, bins: int = 32) -> np.ndarray:
    """Normalised a* and b* channel histograms (64 dims).

    L* (lightness) is excluded because ZSTC grade discrimination is driven
    by chromatic properties (hue and saturation) rather than brightness alone.

    Parameters
    ----------
    lab_img : np.ndarray
        Image in OpenCV L*a*b* format, shape ``(H, W, 3)``, dtype uint8.
    bins : int
        Number of histogram bins per channel (default 32; 2 channels → 64 dims).

    Returns
    -------
    np.ndarray
        Normalised concatenated histogram, shape ``(bins * 2,)``.
    """
    hist_a, _ = np.histogram(
        lab_img[:, :, 1].ravel(), bins=bins, range=(0, 256), density=True
    )
    hist_b, _ = np.histogram(
        lab_img[:, :, 2].ravel(), bins=bins, range=(0, 256), density=True
    )
    return np.concatenate([hist_a, hist_b])


def extract_statistical_moments(lab_img: np.ndarray) -> np.ndarray:
    """Mean, std, and skewness for each L*a*b* channel (9 dims).

    Parameters
    ----------
    lab_img : np.ndarray
        Image in OpenCV L*a*b* format, shape ``(H, W, 3)``.

    Returns
    -------
    np.ndarray
        Shape ``(9,)``: [L_mean, L_std, L_skew, a_mean, a_std, a_skew,
        b_mean, b_std, b_skew].
    """
    features = []
    for c in range(3):
        channel = lab_img[:, :, c].astype(np.float64)
        mean    = np.mean(channel)
        std     = np.std(channel)
        skew    = np.mean((channel - mean) ** 3) / (std ** 3 + 1e-6)
        features.extend([mean, std, skew])
    return np.array(features, dtype=np.float32)


def extract_glcm_features(
    lab_img: np.ndarray,
    distances: List[int] = [1],
    angles: Optional[List[float]] = None,
) -> np.ndarray:
    """Gray-Level Co-occurrence Matrix texture features on L* (20 dims).

    Computes five GLCM properties at 4 angles × 1 distance = 20 values.

    Parameters
    ----------
    lab_img : np.ndarray
        Image in OpenCV L*a*b* format.  Only the L* channel (index 0) is used.
    distances : list[int]
        GLCM inter-pixel distances.
    angles : list[float], optional
        GLCM angles in radians.  Defaults to [0°, 45°, 90°, 135°].

    Returns
    -------
    np.ndarray
        Shape ``(len(distances) * len(angles) * 5,)`` = (1 × 4 × 5) = 20 dims.
    """
    if angles is None:
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]

    l_channel = lab_img[:, :, 0]
    l_ubyte   = img_as_ubyte(l_channel / 255.0)

    features = []
    for dist in distances:
        for angle in angles:
            glcm = graycomatrix(
                l_ubyte,
                distances=[dist],
                angles=[angle],
                levels=256,
                symmetric=True,
                normed=True,
            )
            for prop in ["contrast", "dissimilarity", "homogeneity",
                         "energy", "correlation"]:
                features.append(float(graycoprops(glcm, prop)[0, 0]))

    return np.array(features, dtype=np.float32)


# --------------------------------------------------------------------------- #
# Main extractor
# --------------------------------------------------------------------------- #
def extract_features(image_path: str | Path) -> Optional[np.ndarray]:
    """Extract the full 93-dimensional feature vector from one image.

    Parameters
    ----------
    image_path : str | Path
        Path to a JPEG or PNG image (expected 224×224, RGB).

    Returns
    -------
    np.ndarray or None
        93-dimensional float32 vector, or None if the image could not be
        loaded.
    """
    path = str(image_path)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"[handcrafted_features] Could not read: {path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB)

    hist = extract_colour_histograms(img_lab)     # 64 dims
    stat = extract_statistical_moments(img_lab)   #  9 dims
    glcm = extract_glcm_features(img_lab)         # 20 dims

    return np.concatenate([hist, stat, glcm]).astype(np.float32)


def extract_dataset(
    image_paths: List[str | Path],
) -> Tuple[np.ndarray, List[int]]:
    """Batch-extract features from a list of image paths.

    Parameters
    ----------
    image_paths : list[str | Path]
        Ordered list of image file paths.

    Returns
    -------
    X : np.ndarray
        Feature matrix of shape ``(N_valid, 93)``.
    valid_indices : list[int]
        Indices (into *image_paths*) for which extraction succeeded.
        Use this to align with a parallel labels list.
    """
    features, valid_indices = [], []
    for i, path in enumerate(image_paths):
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            valid_indices.append(i)

    X = np.stack(features) if features else np.empty((0, 93), dtype=np.float32)
    return X, valid_indices


# --------------------------------------------------------------------------- #
# Ablation helper
# --------------------------------------------------------------------------- #
def feature_subset(
    X: np.ndarray,
    include_hist: bool = True,
    include_stat: bool = True,
    include_glcm: bool = True,
) -> np.ndarray:
    """Select a subset of the 93-dimensional feature vector for ablation.

    Slice indices (fixed by the ordering in ``extract_features``):
    - Histogram: columns 0–63
    - Statistical: columns 64–72
    - GLCM: columns 73–92

    Parameters
    ----------
    X : np.ndarray
        Full (N, 93) feature matrix.
    include_hist, include_stat, include_glcm : bool
        Which descriptor groups to retain.

    Returns
    -------
    np.ndarray
        Feature matrix with only the selected columns.
    """
    cols = []
    if include_hist:
        cols.extend(range(0, 64))
    if include_stat:
        cols.extend(range(64, 73))
    if include_glcm:
        cols.extend(range(73, 93))
    if not cols:
        raise ValueError("At least one descriptor group must be selected.")
    return X[:, cols]
