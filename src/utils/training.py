"""
training.py
-----------
Reusable training and validation loops for PyTorch models.

Implements the training configuration used across the 22-architecture
benchmark (notebooks 02–11):

- Optimiser: AdamW (weight decay 1e-4)
- LR schedule: cosine annealing with warm restarts (T_0=10)
- Early stopping: patience 10 on val macro-F1
- Gradient clipping: max_norm 1.0

Reference
---------
Loshchilov, I., & Hutter, F. (2019). Decoupled weight decay regularization.
ICLR 2019. https://arxiv.org/abs/1711.05101
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from .evaluation import macro_f1


# --------------------------------------------------------------------------- #
# Early stopping
# --------------------------------------------------------------------------- #
class EarlyStopping:
    """Stops training when a monitored metric has not improved for *patience*
    consecutive epochs.  Optionally saves the best model checkpoint.

    Parameters
    ----------
    patience : int
        Number of epochs with no improvement before stopping.
    min_delta : float
        Minimum change to qualify as an improvement.
    mode : str
        ``"max"`` (higher is better, e.g., accuracy / F1) or
        ``"min"`` (lower is better, e.g., loss).
    checkpoint_path : str | Path, optional
        If provided, the best model ``state_dict`` is saved here.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 1e-4,
        mode: str = "max",
        checkpoint_path: Optional[str | Path] = None,
    ) -> None:
        self.patience         = patience
        self.min_delta        = min_delta
        self.mode             = mode
        self.checkpoint_path  = Path(checkpoint_path) if checkpoint_path else None
        self.counter          = 0
        self.best_score: Optional[float] = None
        self.stop             = False

    def __call__(self, score: float, model: nn.Module) -> bool:
        """Update internal state.  Returns True when training should stop."""
        improved = (
            self.best_score is None or
            (self.mode == "max" and score > self.best_score + self.min_delta) or
            (self.mode == "min" and score < self.best_score - self.min_delta)
        )
        if improved:
            self.best_score = score
            self.counter    = 0
            if self.checkpoint_path:
                self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(model.state_dict(), self.checkpoint_path)
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return self.stop


# --------------------------------------------------------------------------- #
# Single epoch loops
# --------------------------------------------------------------------------- #
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimiser: torch.optim.Optimizer,
    device: torch.device,
    clip_grad: float = 1.0,
) -> Tuple[float, float]:
    """Run one training epoch.

    Returns
    -------
    avg_loss : float
    avg_accuracy : float
    """
    model.train()
    total_loss, total_correct, total_samples = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimiser.zero_grad()
        outputs = model(inputs)
        loss    = criterion(outputs, targets)
        loss.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        optimiser.step()

        total_loss    += loss.item() * inputs.size(0)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)

    avg_loss = total_loss    / total_samples
    avg_acc  = total_correct / total_samples
    return avg_loss, avg_acc


@torch.no_grad()
def validate_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Run one validation epoch.

    Returns
    -------
    avg_loss : float
    avg_accuracy : float
    val_macro_f1 : float
    """
    model.eval()
    total_loss, total_correct, total_samples = 0.0, 0, 0
    all_preds, all_targets = [], []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss    = criterion(outputs, targets)

        total_loss    += loss.item() * inputs.size(0)
        preds          = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += inputs.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss    / total_samples
    avg_acc  = total_correct / total_samples
    f1       = macro_f1(all_targets, all_preds)
    return avg_loss, avg_acc, f1


# --------------------------------------------------------------------------- #
# Full training loop
# --------------------------------------------------------------------------- #
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    patience: int = 10,
    checkpoint_path: str | Path = "outputs/best_model.pth",
    device: Optional[torch.device] = None,
    log_interval: int = 1,
) -> Dict[str, list]:
    """Full AdamW + cosine annealing training loop with early stopping.

    This is the training configuration used for all PyTorch benchmark models
    (ResNet, DenseNet, EfficientNet, ViT, DeiT, Swin) in the thesis.

    Parameters
    ----------
    model : nn.Module
        Model to train.  Already moved to *device* by the caller.
    train_loader, val_loader : DataLoader
    num_epochs : int
    learning_rate : float
        Initial AdamW learning rate (default 1e-4).
    weight_decay : float
        AdamW weight decay (default 1e-4).
    patience : int
        EarlyStopping patience on val macro-F1.
    checkpoint_path : str | Path
        Where to save the best model state_dict.
    device : torch.device, optional
        Defaults to CUDA if available.
    log_interval : int
        Print a log line every *log_interval* epochs.

    Returns
    -------
    history : dict
        Keys: ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``,
        ``val_f1`` — each a list of per-epoch values.
    """
    device = device or (
        torch.device("cuda") if torch.cuda.is_available()
        else torch.device("cpu")
    )
    model.to(device)

    criterion  = nn.CrossEntropyLoss()
    optimiser  = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler  = CosineAnnealingWarmRestarts(optimiser, T_0=10, T_mult=1)
    stopper    = EarlyStopping(
        patience=patience, mode="max",
        checkpoint_path=checkpoint_path,
    )

    history: Dict[str, list] = {
        "train_loss": [], "train_acc": [],
        "val_loss":   [], "val_acc":   [], "val_f1": [],
    }

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_one_epoch(
            model, train_loader, criterion, optimiser, device
        )
        vl_loss, vl_acc, vl_f1 = validate_one_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step(epoch)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(vl_loss)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        if epoch % log_interval == 0:
            elapsed = time.time() - t0
            print(
                f"Epoch [{epoch:3d}/{num_epochs}] "
                f"| train loss={tr_loss:.4f} acc={tr_acc*100:.2f}% "
                f"| val   loss={vl_loss:.4f} acc={vl_acc*100:.2f}% "
                f"F1={vl_f1*100:.2f}% "
                f"| {elapsed:.1f}s"
            )

        if stopper(vl_f1, model):
            print(f"Early stopping at epoch {epoch} "
                  f"(best val F1={stopper.best_score*100:.2f}%)")
            break

    # Restore best weights
    if Path(checkpoint_path).exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Best weights restored from {checkpoint_path}")

    return history
