"""
clove_dataset.py
----------------
PyTorch Dataset for single-clove images (Grade I–IV).

Folder layout expected::

    root/
    ├── Grade 1/
    ├── Grade 2/
    ├── Grade 3/
    └── Grade 4/

Images are loaded as 224×224 RGB tensors.  The optional ``transform``
argument accepts any torchvision transform pipeline (default: ToTensor +
ImageNet normalisation).

Used in: notebooks/02–08 (benchmark), notebook/11 (context-aware ResNet-18).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GRADE_FOLDERS: dict[str, int] = {
    "Grade 1": 0,
    "Grade 2": 1,
    "Grade 3": 2,
    "Grade 4": 3,
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2,
                           saturation=0.2, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class CloveDataset(Dataset):
    """Single-clove image dataset aligned with ZSTC Grade I–IV labels.

    Parameters
    ----------
    root : str | Path
        Root directory containing one sub-folder per grade.
    transform : callable, optional
        Transform applied to each PIL image.  Defaults to
        ``DEFAULT_TRANSFORM`` (resize 224×224, ToTensor, ImageNet
        normalisation).
    split : str, optional
        If ``"train"``, applies ``TRAIN_TRANSFORM`` augmentations unless
        *transform* is explicitly provided.  Ignored when *transform* is set.
    extensions : tuple[str, ...]
        Valid image file extensions (case-insensitive).
    """

    def __init__(
        self,
        root: str | Path,
        transform: Optional[Callable] = None,
        split: str = "test",
        extensions: Tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp"),
    ) -> None:
        self.root = Path(root)
        self.extensions = extensions
        self.classes = list(GRADE_FOLDERS.keys())
        self.class_to_idx = GRADE_FOLDERS

        # Choose transform
        if transform is not None:
            self.transform = transform
        elif split == "train":
            self.transform = TRAIN_TRANSFORM
        else:
            self.transform = DEFAULT_TRANSFORM

        self.samples: list[Tuple[Path, int]] = []
        self._load_samples()

    # ------------------------------------------------------------------
    def _load_samples(self) -> None:
        for grade_name, label in GRADE_FOLDERS.items():
            folder = self.root / grade_name
            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Grade folder not found: {folder}\n"
                    f"Expected sub-folders: {list(GRADE_FOLDERS.keys())}"
                )
            for f in sorted(folder.iterdir()):
                if f.suffix.lower() in self.extensions:
                    self.samples.append((f, label))

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No images found under '{self.root}'. "
                f"Check that folders are named 'Grade 1', 'Grade 2', etc."
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        per_class = {g: 0 for g in GRADE_FOLDERS}
        for _, lbl in self.samples:
            grade = self.classes[lbl]
            per_class[grade] += 1
        lines = [f"CloveDataset(root={self.root}, n={len(self)})"]
        for g, n in per_class.items():
            lines.append(f"  {g}: {n} images")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Convenience factory
# --------------------------------------------------------------------------- #
def build_dataloaders(
    train_root: str | Path,
    val_root: str | Path,
    test_root: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
    torch.utils.data.DataLoader,
]:
    """Return (train_loader, val_loader, test_loader) for the clove dataset.

    Parameters
    ----------
    train_root, val_root, test_root :
        Paths to the pre-split train / val / test directories (each with
        Grade 1–4 sub-folders).
    batch_size : int
        Mini-batch size for the DataLoader.
    num_workers : int
        Number of DataLoader worker processes.

    Returns
    -------
    Tuple of (train_loader, val_loader, test_loader).
    """
    from torch.utils.data import DataLoader

    train_ds = CloveDataset(train_root, split="train")
    val_ds   = CloveDataset(val_root,   split="val")
    test_ds  = CloveDataset(test_root,  split="test")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return train_loader, val_loader, test_loader
