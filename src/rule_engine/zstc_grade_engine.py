"""
zstc_grade_engine.py
--------------------
Deterministic ZSTC threshold logic and audit trail generation.

This module encodes the official Zanzibar State Trading Corporation (ZSTC)
clove grading standard as deterministic program logic — not as a training
objective approximated by gradient descent.  Every grade decision produced
by this module is accompanied by a human-readable audit trail that can be
independently verified by a ZSTC officer without any specialised AI tools.

ZSTC Official Grading Thresholds (source: ZSTC Internal Circular, 2023)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
┌──────────┬──────────────────────────────────┬────────────┬──────────────┐
│  Grade   │  Primary Attribute               │ Mpeta (≤)  │ F. Matter (≤)│
├──────────┼──────────────────────────────────┼────────────┼──────────────┤
│ Grade I  │ Attractive golden colour         │    3%      │      5%      │
│ Grade II │ Faded / blackish colour          │    7%      │      5%      │
│ Grade III│ More faded colour                │   20%      │      5%      │
│ Grade IV │ Primarily mpeta / fermented      │   >20%     │     N/A      │
└──────────┴──────────────────────────────────┴────────────┴──────────────┘

Mpeta fraction (γ) is defined as:
    γ = count(predicted mpeta instances) / count(all clove instances)

Two modes
~~~~~~~~~
Single-clove mode:
    ``assign_grade(grade_logits)`` aggregates per-instance predictions
    across a batch to compute γ and applies the threshold ladder.

Batch-clove mode:
    ``lookup_grade(pred_class)`` maps the EfficientNet-Lite0 predicted
    class index directly to a ZSTC grade and threshold statement.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch

# --------------------------------------------------------------------------- #
# Constants — ZSTC thresholds
# --------------------------------------------------------------------------- #
GRADE_NAMES = ["Grade I", "Grade II", "Grade III", "Grade IV"]

# Mpeta fraction upper bounds for Grades I, II, III.  Grade IV = above Grade III.
MPETA_THRESHOLDS: Dict[str, float] = {
    "Grade I":   0.03,
    "Grade II":  0.07,
    "Grade III": 0.20,
}

# Class index used by the context-aware ResNet-18 for "mpeta" instances
# (Grade IV label at the per-clove level).
MPETA_CLASS_IDX = 3   # index 3 = "Grade 4" (= mpeta-dominated clove)

# EfficientNet-Lite0 batch-mode class map (must match batch_dataset.py)
BATCH_CLASS_NAMES = ["Grade_1", "Grade_2", "Grade_3", "Grade_4", "Not_Clove"]
BATCH_TO_ZSTC: Dict[int, Optional[str]] = {
    0: "Grade I",
    1: "Grade II",
    2: "Grade III",
    3: "Grade IV",
    4: None,           # Not_Clove → rejection
}


# --------------------------------------------------------------------------- #
# Audit record dataclass
# --------------------------------------------------------------------------- #
@dataclass
class AuditRecord:
    """Structured audit trail for a single ZSTC grading decision.

    Attributes
    ----------
    grade : str or None
        Assigned ZSTC grade ("Grade I" … "Grade IV") or None if the input
        was rejected as not-a-clove.
    mpeta_fraction : float or None
        Computed or inferred mpeta fraction γ.  None for batch-mode
        decisions that do not compute γ explicitly.
    threshold_applied : str
        Human-readable statement of the ZSTC threshold rule that was applied.
    mode : str
        Either ``"single_clove"`` or ``"batch_clove"``.
    n_total : int
        Total number of clove instances examined (single-clove mode only).
    n_mpeta : int
        Number of mpeta instances (single-clove mode only).
    timestamp : str
        ISO-8601 timestamp of the decision.
    raw_counts : dict
        Per-grade instance counts (single-clove mode).
    """
    grade: Optional[str]
    mpeta_fraction: Optional[float]
    threshold_applied: str
    mode: str = "single_clove"
    n_total: int = 0
    n_mpeta: int = 0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    raw_counts: Dict[str, int] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def to_text(self) -> str:
        """Return the human-readable audit trail string."""
        if self.grade is None:
            return (
                f"[{self.timestamp}] REJECTION: Input not classified as clove. "
                f"Threshold: {self.threshold_applied}"
            )
        frac_str = (
            f"{self.mpeta_fraction * 100:.1f}%"
            if self.mpeta_fraction is not None else "N/A"
        )
        return (
            f"[{self.timestamp}] "
            f"{self.grade} assigned. "
            f"Mpeta = {frac_str}. "
            f"Threshold applied: {self.threshold_applied}."
        )

    def to_dict(self) -> dict:
        """Return the audit record as a JSON-serialisable dictionary."""
        return {
            "grade":              self.grade,
            "mpeta_fraction":     self.mpeta_fraction,
            "threshold_applied":  self.threshold_applied,
            "mode":               self.mode,
            "n_total":            self.n_total,
            "n_mpeta":            self.n_mpeta,
            "timestamp":          self.timestamp,
            "raw_counts":         self.raw_counts,
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialise the audit record as a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


# --------------------------------------------------------------------------- #
# Grade engine
# --------------------------------------------------------------------------- #
class ZSTCGradeEngine:
    """Deterministic ZSTC grading engine with audit trail generation.

    This class encodes the ZSTC threshold ladder as explicit conditional
    logic.  It accepts outputs from both the context-aware ResNet-18
    (single-clove mode) and EfficientNet-Lite0 (batch-clove mode) and
    produces a structured AuditRecord for every decision.

    Usage — Single-Clove Mode
    --------------------------
    Collect per-instance grade predictions across all cloves in a batch
    image, then call ``assign_grade``:

    >>> engine = ZSTCGradeEngine()
    >>> logits = cls_model(x_batch)           # (N_instances, 4)
    >>> grade, audit = engine.assign_grade(logits)
    >>> print(audit.to_text())
    [2026-04-15T08:23:11] Grade II assigned. Mpeta = 5.3%. Threshold: mpeta <= 7%.

    Usage — Batch-Clove Mode
    -------------------------
    >>> pred_class = 1                        # EfficientNet-Lite0 output
    >>> grade, audit = engine.lookup_grade(pred_class)
    >>> print(audit.to_text())
    [2026-04-15T08:23:11] Grade II assigned. Mpeta = N/A. Threshold: ...
    """

    # ------------------------------------------------------------------
    def assign_grade(
        self,
        grade_logits: Union[torch.Tensor, np.ndarray],
        mpeta_class_idx: int = MPETA_CLASS_IDX,
    ) -> Tuple[Optional[str], AuditRecord]:
        """Compute mpeta fraction and assign ZSTC grade (single-clove mode).

        Parameters
        ----------
        grade_logits : torch.Tensor or np.ndarray
            Per-instance logits or probabilities of shape ``(N, 4)`` where N
            is the number of detected clove instances in the image.
        mpeta_class_idx : int
            Class index considered as mpeta (default 3 = Grade IV).

        Returns
        -------
        grade : str
            Assigned ZSTC grade ("Grade I" … "Grade IV").
        audit : AuditRecord
            Structured audit trail for this decision.
        """
        if isinstance(grade_logits, torch.Tensor):
            preds = grade_logits.argmax(dim=1).cpu().numpy()
        else:
            preds = np.argmax(grade_logits, axis=1)

        n_total = len(preds)
        if n_total == 0:
            rec = AuditRecord(
                grade=None,
                mpeta_fraction=None,
                threshold_applied="No clove instances detected.",
                mode="single_clove",
            )
            return None, rec

        # Per-grade counts for the audit record
        raw_counts = {f"Grade {i+1}": int(np.sum(preds == i)) for i in range(4)}
        n_mpeta    = int(np.sum(preds == mpeta_class_idx))
        gamma      = n_mpeta / n_total

        # Threshold ladder (ZSTC order: I → II → III → IV)
        if gamma <= MPETA_THRESHOLDS["Grade I"]:
            grade = "Grade I"
            thr   = f"mpeta \u2264 {MPETA_THRESHOLDS['Grade I']*100:.0f}%"
        elif gamma <= MPETA_THRESHOLDS["Grade II"]:
            grade = "Grade II"
            thr   = (
                f"{MPETA_THRESHOLDS['Grade I']*100:.0f}% < mpeta "
                f"\u2264 {MPETA_THRESHOLDS['Grade II']*100:.0f}%"
            )
        elif gamma <= MPETA_THRESHOLDS["Grade III"]:
            grade = "Grade III"
            thr   = (
                f"{MPETA_THRESHOLDS['Grade II']*100:.0f}% < mpeta "
                f"\u2264 {MPETA_THRESHOLDS['Grade III']*100:.0f}%"
            )
        else:
            grade = "Grade IV"
            thr   = f"mpeta > {MPETA_THRESHOLDS['Grade III']*100:.0f}%"

        audit = AuditRecord(
            grade=grade,
            mpeta_fraction=gamma,
            threshold_applied=thr,
            mode="single_clove",
            n_total=n_total,
            n_mpeta=n_mpeta,
            raw_counts=raw_counts,
        )
        return grade, audit

    # ------------------------------------------------------------------
    def lookup_grade(
        self,
        pred_class: int,
    ) -> Tuple[Optional[str], AuditRecord]:
        """Map a batch-mode classifier prediction to a ZSTC grade.

        The EfficientNet-Lite0 batch classifier predicts one of five classes
        (Grade_1 … Grade_4, Not_Clove).  This method maps that prediction
        to the corresponding ZSTC grade and generates the audit trail.

        Note: the mpeta fraction is not computed in batch mode (pile-level
        classification cannot isolate individual instances).  The audit trail
        states the ZSTC threshold bracket that corresponds to the predicted
        grade, as a practical approximation.

        Parameters
        ----------
        pred_class : int
            Predicted class index from EfficientNet-Lite0 (0–4).

        Returns
        -------
        grade : str or None
            ZSTC grade, or None if the input was rejected (Not_Clove).
        audit : AuditRecord
        """
        grade = BATCH_TO_ZSTC.get(pred_class)

        if grade is None:
            thr = "Input classified as Not_Clove (rejection class)."
        elif grade == "Grade I":
            thr = f"mpeta \u2264 {MPETA_THRESHOLDS['Grade I']*100:.0f}% (pile-level estimate)"
        elif grade == "Grade II":
            thr = (
                f"{MPETA_THRESHOLDS['Grade I']*100:.0f}% < mpeta "
                f"\u2264 {MPETA_THRESHOLDS['Grade II']*100:.0f}% (pile-level estimate)"
            )
        elif grade == "Grade III":
            thr = (
                f"{MPETA_THRESHOLDS['Grade II']*100:.0f}% < mpeta "
                f"\u2264 {MPETA_THRESHOLDS['Grade III']*100:.0f}% (pile-level estimate)"
            )
        else:
            thr = f"mpeta > {MPETA_THRESHOLDS['Grade III']*100:.0f}% (pile-level estimate)"

        audit = AuditRecord(
            grade=grade,
            mpeta_fraction=None,
            threshold_applied=thr,
            mode="batch_clove",
        )
        return grade, audit

    # ------------------------------------------------------------------
    def save_audit_log(
        self,
        records: List[AuditRecord],
        output_path: str | Path,
    ) -> None:
        """Serialise a list of audit records to a JSON file.

        Parameters
        ----------
        records : list[AuditRecord]
        output_path : str | Path
            Destination ``.json`` file.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump([r.to_dict() for r in records], f, indent=2)
        print(f"[ZSTCGradeEngine] Saved {len(records)} audit records → {output_path}")
