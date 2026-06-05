"""
mask_dataset.py
---------------
4-channel RGB + binary-mask dataset loader for the context-aware ResNet-18.

Each sample is a (4, 224, 224) tensor: the standard 3-channel RGB image
concatenated with a single-channel binary segmentation mask produced by
YOLOv8-seg.  The mask channel conditions the classifier on the precise
spatial extent of the clove, suppressing background features that may
confound grade prediction.

Weight initialisation note
~~~~~~~~~~~~~~~~~~~~~~~~~~
The 4th input channel of ResNet-18's Conv1 is initialised to the
channel-mean of the pre-trained RGB weights::

    conv1.weight[:, 3:4] = conv1.weight.mean(dim=1, keepdim=True)

This ensures that a mask of all-ones (full-image receptive field) recovers
the original 3-channel behaviour at initialisation.

Folder layout expected::

    images_root/
    ├── Grade 1/   ← RGB images (224×224)
    ├── Grade 2/
    ├── Grade 3/
    └── Grade 4/

    masks_root/
    ├── Grade 1/   ← binary PNG masks (same filename stem as RGB)
    ├── Grade 2/
    ├── Grade 3/
    └── Grade 4/

If a mask file is missing for a given image, a full-ones mask is substituted
so the module degrades gracefully to a standard 3-channel classifier.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from .clove_dataset import GRADE_FOLDERS, IMAGENET_MEAN, IMAGENET_STD

# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #
_RGB_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

_MASK_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),  # → [0, 1] float, shape (1, H, W)
])

_RGB_TRAIN = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.03),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #
class MaskDataset(Dataset):
    """4-channel RGB + binary-mask dataset for context-aware classification.

    Parameters
    ----------
    images_root : str | Path
        Root folder of RGB images (Grade 1 … Grade 4 sub-folders).
    masks_root : str | Path
        Root folder of corresponding binary masks.  Mask filenames must
        share the stem with their paired RGB image (e.g. ``clove_001.png``
        pairs with ``clove_001.jpg``).
    split : str
        ``"train"`` applies data augmentation.  Any other value uses the
        deterministic evaluation transform.
    rgb_transform : callable, optional
        Override the default RGB transform.
    mask_transform : callable, optional
        Override the default mask transform.
    """

    MASK_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        images_root: str | Path,
        masks_root: str | Path,
        split: str = "test",
        rgb_transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ) -> None:
        self.images_root = Path(images_root)
        self.masks_root  = Path(masks_root)
        self.split       = split

        self.rgb_transform  = rgb_transform  or (_RGB_TRAIN if split == "train" else _RGB_TRANSFORM)
        self.mask_transform = mask_transform or _MASK_TRANSFORM

        self.samples: list[Tuple[Path, Optional[Path], int]] = []
        self._load_samples()

    # ------------------------------------------------------------------
    def _find_mask(self, grade_folder: Path, stem: str) -> Optional[Path]:
        for ext in self.MASK_EXTENSIONS:
            candidate = grade_folder / (stem + ext)
            if candidate.exists():
                return candidate
        return None

    def _load_samples(self) -> None:
        img_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        for grade_name, label in GRADE_FOLDERS.items():
            img_dir  = self.images_root / grade_name
            mask_dir = self.masks_root  / grade_name
            if not img_dir.is_dir():
                raise FileNotFoundError(f"Image folder not found: {img_dir}")
            for img_path in sorted(img_dir.iterdir()):
                if img_path.suffix.lower() not in img_exts:
                    continue
                mask_path = self._find_mask(mask_dir, img_path.stem)
                self.samples.append((img_path, mask_path, label))

        missing = sum(1 for _, m, _ in self.samples if m is None)
        if missing:
            print(
                f"[MaskDataset] Warning: {missing}/{len(self.samples)} images "
                f"have no paired mask — full-ones mask substituted."
            )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, mask_path, label = self.samples[idx]

        # RGB → (3, 224, 224)
        rgb   = Image.open(img_path).convert("RGB")
        rgb_t = self.rgb_transform(rgb)

        # Mask → (1, 224, 224); fall back to all-ones if missing
        if mask_path is not None:
            mask   = Image.open(mask_path).convert("L")
            mask_t = self.mask_transform(mask)
            mask_t = (mask_t > 0.5).float()          # binarise
        else:
            mask_t = torch.ones(1, 224, 224)

        # Concatenate → (4, 224, 224)
        x = torch.cat([rgb_t, mask_t], dim=0)
        return x, label

    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        n_masked = sum(1 for _, m, _ in self.samples if m is not None)
        return (
            f"MaskDataset(n={len(self)}, "
            f"masked={n_masked}, "
            f"split='{self.split}')"
        )
