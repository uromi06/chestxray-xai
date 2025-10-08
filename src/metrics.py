"""
metrics.py
----------
Utilities for operating-point selection, confusion matrix reporting,
balanced accuracy, and optional calibration plotting.
"""

from __future__ import annotations
import numpy as np
import torch
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
from typing import Dict, Tuple


# ---------- core helpers ----------

@torch.no_grad()
def logits_labels(model, loader, device="cpu") -> Tuple[np.ndarray, np.ndarray]:
    """Collect raw logits and labels from a loader."""
    model.eval()
    all_logits, all_y = [], []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).detach().cpu().numpy()
        all_logits.append(logits.reshape(-1))
        all_y.append(y.numpy().astype(int))
    return np.concatenate(all_logits), np.concatenate(all_y)

def proba_from_logits(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits))

def confusion_at_threshold(y_true: np.ndarray, p: np.ndarray, thr: float) -> Dict[str, float]:
    """Compute confusion-derived metrics at a chosen threshold."""
    y_pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sens = tp / (tp + fn + 1e-9)      # recall for positives
    spec = tn / (tn + fp + 1e-9)      # recall for negatives
    prec = tp / (tp + fp + 1e-9)
    acc  = (tp + tn) / (tp + tn + fp + fn + 1e-9)
    bal_acc = 0.5 * (sens + spec)
    f1 = 2 * prec * sens / (prec + sens + 1e-9)
    return dict(tn=tn, fp=fp, fn=fn, tp=tp,
                sensitivity=sens, specificity=spec,
                precision=prec, accuracy=acc,
                balanced_accuracy=bal_acc, f1=f1)

def scan_thresholds(y_true: np.ndarray, p: np.ndarray, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    rows = []
    for t in grid:
        m = confusion_at_threshold(y_true, p, t)
        m["thr"] = float(t)
        rows.append(m)
    return rows


# ---------- operating point pickers ----------

def pick_screening(y_true: np.ndarray, p: np.ndarray, min_sens: float = 0.98) -> float:
    """
    Choose the *highest* specificity threshold among those with sensitivity >= min_sens.
    Good for screening (don't miss disease).
    """
    rows = scan_thresholds(y_true, p)
    candidates = [r for r in rows if r["sensitivity"] >= min_sens]
    if not candidates:
        # fallback: just return threshold with max sensitivity
        return max(rows, key=lambda r: r["sensitivity"])["thr"]
    # among candidates, pick the one with best specificity (ties -> higher thr)
    candidates.sort(key=lambda r: (r["specificity"], r["thr"]))
    return candidates[-1]["thr"]

def pick_rulein(y_true: np.ndarray, p: np.ndarray, min_spec: float = 0.85) -> float:
    """
    Choose threshold that achieves at least min_spec and maximizes precision.
    Good for rule-in (high PPV / specificity).
    """
    rows = scan_thresholds(y_true, p)
    candidates = [r for r in rows if r["specificity"] >= min_spec]
    if not candidates:
        # fallback: threshold with max specificity
        return max(rows, key=lambda r: r["specificity"])["thr"]
    candidates.sort(key=lambda r: (r["precision"], r["thr"]))
    return candidates[-1]["thr"]


# ---------- summary print ----------

def print_report(y_true: np.ndarray, p: np.ndarray, thr: float, header="TEST"):
    auroc = roc_auc_score(y_true, p)
    auprc = average_precision_score(y_true, p)
    m = confusion_at_threshold(y_true, p, thr)
    print(f"{header}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  Thr={thr:.2f}  "
          f"Sens={m['sensitivity']:.3f}  Spec={m['specificity']:.3f}  "
          f"Prec={m['precision']:.3f}  Acc={m['accuracy']:.3f}  "
          f"BalAcc={m['balanced_accuracy']:.3f}  F1={m['f1']:.3f}")
    print(f"Confusion  TP={m['tp']}  FP={m['fp']}  TN={m['tn']}  FN={m['fn']}")


# ---------- optional: temperature scaling + reliability curve ----------

class TemperatureScaler(torch.nn.Module):
    """
    Platt-style temperature scaling for logits. Fit by minimizing NLL on validation logits.
    """
    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = torch.nn.Parameter(torch.log(torch.tensor([init_T], dtype=torch.float32)))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        T = torch.exp(self.log_T)
        return logits / T

    def fit(self, logits_val: np.ndarray, y_val: np.ndarray, max_iter: int = 200):
        self.train()
        # to tensors
        z = torch.tensor(logits_val, dtype=torch.float32).unsqueeze(1)   # [N,1]
        y = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
        opt = torch.optim.LBFGS([self.log_T], lr=0.1, max_iter=max_iter)

        loss_fn = torch.nn.BCEWithLogitsLoss()

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(self(z), y)
            loss.backward()
            return loss

        opt.step(closure)
        return float(torch.exp(self.log_T).item())

def reliability_curve(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10, title="Reliability"):
    """Plot calibration curve (no return; shows a figure)."""
    import sklearn.calibration as cal
    frac_pos, mean_pred = cal.calibration_curve(y_true, p, n_bins=n_bins, strategy="uniform")
    plt.figure()
    plt.plot([0,1], [0,1], linestyle="--", label="Perfect")
    plt.plot(mean_pred, frac_pos, marker="o", label="Model")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
