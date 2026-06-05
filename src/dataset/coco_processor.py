"""
coco_processor.py
-----------------
Convert CVAT-exported COCO JSON polygon annotations to per-instance binary
PNG masks for use with YOLOv8-seg and the MaskDataset loader.

The 200 single-clove images annotated in Roboflow (polygon masks, COCO format)
were exported as a COCO JSON file.  This module converts that JSON into the
per-image binary mask files consumed by MaskDataset and the YOLOv8-seg
training pipeline.

Usage
-----
From the command line::

    python -m src.dataset.coco_processor \\
        --json   annotations/instances_train.json \\
        --images data/images/train \\
        --output data/masks/train

Or programmatically::

    from src.dataset.coco_processor import CocoMaskProcessor
    proc = CocoMaskProcessor("annotations/instances_train.json")
    proc.export_masks("data/images/train", "data/masks/train")

Output
------
One binary PNG mask per image, saved to ``output/<image_filename_stem>.png``.
Each mask is 0 (background) or 255 (clove instance).  If multiple instances
are present in one image, all are merged into a single binary mask.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
# Core processor
# --------------------------------------------------------------------------- #
class CocoMaskProcessor:
    """Rasterise COCO polygon annotations to binary masks.

    Parameters
    ----------
    json_path : str | Path
        Path to the COCO-format JSON annotation file.
    """

    def __init__(self, json_path: str | Path) -> None:
        self.json_path = Path(json_path)
        with open(self.json_path) as f:
            self._coco = json.load(f)
        self._index()

    # ------------------------------------------------------------------
    def _index(self) -> None:
        """Build fast lookup dicts from the COCO JSON."""
        self._images: Dict[int, dict] = {
            img["id"]: img for img in self._coco["images"]
        }
        # Group annotations by image_id
        self._anns_by_image: Dict[int, List[dict]] = {}
        for ann in self._coco.get("annotations", []):
            img_id = ann["image_id"]
            self._anns_by_image.setdefault(img_id, []).append(ann)

    # ------------------------------------------------------------------
    def _rasterise(self, img_meta: dict, annotations: List[dict]) -> np.ndarray:
        """Rasterise all polygon annotations for one image.

        Parameters
        ----------
        img_meta : dict
            COCO image dict (keys: ``width``, ``height``, ``file_name``).
        annotations : list[dict]
            COCO annotation dicts for this image.  Polygon segmentation
            assumed (``segmentation`` is a list of flat [x1,y1,x2,y2,…]).

        Returns
        -------
        np.ndarray
            Binary mask of shape ``(H, W)``, dtype ``uint8``, values 0/255.
        """
        H, W = img_meta["height"], img_meta["width"]
        mask = np.zeros((H, W), dtype=np.uint8)

        for ann in annotations:
            segs = ann.get("segmentation", [])
            if isinstance(segs, dict):
                # RLE format — decode via pycocotools if available
                try:
                    from pycocotools import mask as coco_mask
                    rle = coco_mask.frPyObjects(segs, H, W)
                    m   = coco_mask.decode(rle)
                    mask = np.maximum(mask, (m * 255).astype(np.uint8))
                except ImportError:
                    print("[coco_processor] pycocotools not installed; "
                          "RLE masks skipped.")
                continue

            for polygon in segs:
                pts = np.array(polygon, dtype=np.float32).reshape(-1, 2)
                pts = pts.astype(np.int32)
                cv2.fillPoly(mask, [pts], 255)

        return mask

    # ------------------------------------------------------------------
    def export_masks(
        self,
        images_dir: str | Path,
        output_dir: str | Path,
        missing_ok: bool = True,
    ) -> None:
        """Write one binary PNG mask per annotated image.

        Parameters
        ----------
        images_dir : str | Path
            Directory containing the source images (used only to resolve
            relative ``file_name`` paths in the COCO JSON).
        output_dir : str | Path
            Destination directory for the PNG masks.
        missing_ok : bool
            If True, images present in *images_dir* but absent from the
            annotation JSON are silently skipped.
        """
        images_dir = Path(images_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        n_written = 0
        for img_id, img_meta in self._images.items():
            annotations = self._anns_by_image.get(img_id, [])
            if not annotations:
                if not missing_ok:
                    raise ValueError(
                        f"Image id={img_id} ({img_meta['file_name']}) "
                        f"has no annotations."
                    )
                continue

            mask   = self._rasterise(img_meta, annotations)
            stem   = Path(img_meta["file_name"]).stem
            out_fp = output_dir / (stem + ".png")
            Image.fromarray(mask).save(out_fp)
            n_written += 1

        print(f"[coco_processor] Exported {n_written} masks → {output_dir}")

    # ------------------------------------------------------------------
    def summary(self) -> None:
        """Print a brief summary of the loaded COCO annotation file."""
        n_imgs  = len(self._images)
        n_anns  = len(self._coco.get("annotations", []))
        n_cats  = len(self._coco.get("categories", []))
        cats    = [c["name"] for c in self._coco.get("categories", [])]
        print(
            f"COCO JSON: {self.json_path.name}\n"
            f"  Images:      {n_imgs}\n"
            f"  Annotations: {n_anns}\n"
            f"  Categories:  {n_cats} → {cats}"
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert CVAT COCO polygon annotations to binary PNG masks."
    )
    p.add_argument("--json",    required=True, help="Path to COCO JSON file.")
    p.add_argument("--images",  required=True, help="Directory of source images.")
    p.add_argument("--output",  required=True, help="Output directory for masks.")
    p.add_argument(
        "--strict", action="store_true",
        help="Raise an error if an image has no annotations.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    proc = CocoMaskProcessor(args.json)
    proc.summary()
    proc.export_masks(args.images, args.output, missing_ok=not args.strict)
