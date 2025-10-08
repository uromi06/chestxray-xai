"""
eval.py
-------
Evaluation utilities for binary classification (NORMAL vs PNEUMONIA).

What's here:
- collect_probs_labels(): runs the model on a DataLoader and returns predicted
  probabilities and ground-truth labels as numpy arrays.
- evaluate(): computes AUROC, AUPRC, Sensitivity, Specificity at a fixed
  threshold (default 0.5). Useful for quick checks and logging during training.
- find_best_threshold(): sweeps thresholds on a *validation* set to pick the
  operating point that maximizes either Youden's J (Sens + Spec - 1) or F1.
- evaluate_at_threshold(): computes sensitivity/specificity (and confusion)
  at an arbitrary threshold (e.g., the tuned threshold you found on val).

Author: Urmi Bhattacharyya
Date: September 2025
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
)

# ---------------------------------------------------------------------
# Core helper: run model → collect predicted probabilities and labels
# ---------------------------------------------------------------------
@torch.no_grad()
def collect_probs_labels(model: torch.nn.Module,
                         loader: torch.utils.data.DataLoader,
                         device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the model on a dataset and collects sigmoid probabilities and labels.

    Args:
        model: trained model (binary head returning logits of shape [B])
        loader: DataLoader yielding (images, labels)
        device: "cpu" or "cuda"

    Returns:
        probs: float array of shape [N], each in [0, 1]
        labels: int/float array of shape [N], values 0 or 1
    """
    model.eval()
    probs, labels = [], []

    for x, y in loader:
        x = x.to(device)
        y = y.float()  # ensure float for metrics that expect numeric
        logits = model(x)               # [B] raw scores
        p = torch.sigmoid(logits)       # [B] probabilities in [0,1]
        probs.append(p.cpu().numpy())
        labels.append(y.cpu().numpy())

    probs = np.concatenate(probs, axis=0)
    labels = np.concatenate(labels, axis=0)
    return probs, labels


# ---------------------------------------------------------------------
# Quick evaluation at a fixed threshold (default 0.5)
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: torch.nn.Module,
             loader: torch.utils.data.DataLoader,
             device: str = "cpu",
             thr: float = 0.5) -> tuple[float, float, float, float]:
    """
    Computes AUROC/AUPRC (threshold-free) and Sens/Spec at a given threshold.

    Args:
        model: trained model
        loader: DataLoader
        device: "cpu" or "cuda"
        thr: decision threshold for converting probs → class (default 0.5)

    Returns:
        auroc, auprc, sens, spec
    """
    probs, labels = collect_probs_labels(model, loader, device)

    # Threshold-free metrics (recommended to always report)
    auroc = roc_auc_score(labels, probs)
    auprc = average_precision_score(labels, probs)

    # Thresholded metrics (operating point)
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
    sens = tp / (tp + fn + 1e-9)  # recall for positive class (PNEUMONIA)
    spec = tn / (tn + fp + 1e-9)  # recall for negative class (NORMAL)

    return auroc, auprc, sens, spec


# ---------------------------------------------------------------------
# Tune decision threshold on a *validation* set
# ---------------------------------------------------------------------
@torch.no_grad()
def find_best_threshold(model: torch.nn.Module,
                        loader: torch.utils.data.DataLoader,
                        device: str = "cpu",
                        mode: str = "youden") -> float:
    """
    Finds the threshold that maximizes either:
      - Youden's J = Sensitivity + Specificity - 1
      - F1 score    = 2 * precision * recall / (precision + recall)

    Use on the validation loader, then evaluate that threshold on the test set.

    Args:
        model: trained (or current) model
        loader: validation DataLoader
        device: "cpu" or "cuda"
        mode: "youden" or "f1"

    Returns:
        best_thr: float in (0,1) giving best operating point on validation
    """
    assert mode in {"youden", "f1"}, "mode must be 'youden' or 'f1'"

    probs, labels = collect_probs_labels(model, loader, device)

    thresholds = np.linspace(0.01, 0.99, 99)
    best_thr, best_score = 0.5, -np.inf

    for t in thresholds:
        pred = (probs >= t).astype(int)

        if mode == "youden":
            tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()
            sens = tp / (tp + fn + 1e-9)
            spec = tn / (tn + fp + 1e-9)
            score = sens + spec - 1.0  # Youden's J
        else:  # mode == "f1"
            score = f1_score(labels, pred)

        if score > best_score:
            best_score = score
            best_thr = float(t)

    return best_thr


# ---------------------------------------------------------------------
# Evaluate sens/spec at a chosen threshold (e.g., tuned on validation)
# ---------------------------------------------------------------------
@torch.no_grad()
def evaluate_at_threshold(model: torch.nn.Module,
                          loader: torch.utils.data.DataLoader,
                          thr: float,
                          device: str = "cpu") -> dict:
    """
    Computes a full set of confusion-derived metrics at a specific threshold.

    Args:
        model: trained model
        loader: DataLoader to evaluate on (val or test)
        thr: threshold in (0,1) to binarize probabilities
        device: "cpu" or "cuda"

    Returns:
        metrics dict with:
            - 'threshold'
            - 'tn', 'fp', 'fn', 'tp'
            - 'sensitivity' (recall for positive)
            - 'specificity' (recall for negative)
            - 'precision'   (tp / (tp + fp))
            - 'accuracy'
    """
    probs, labels = collect_probs_labels(model, loader, device)
    pred = (probs >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, pred).ravel()

    sens = tp / (tp + fn + 1e-9)  # TPR / recall+
    spec = tn / (tn + fp + 1e-9)  # TNR / recall-
    prec = tp / (tp + fp + 1e-9)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-9)

    return {
        "threshold": thr,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "sensitivity": float(sens),
        "specificity": float(spec),
        "precision": float(prec),
        "accuracy": float(acc),
    }
