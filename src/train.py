<<<<<<< HEAD
"""
train.py — robust + user-friendly (final calibrated version)
"""

from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .data_folder import make_loaders_folder
from .model import ResNet18Bin
from .eval import evaluate, find_best_threshold, evaluate_at_threshold
from src.metrics import (
    TemperatureScaler, logits_labels, proba_from_logits, reliability_curve
)

# ------------------------- reproducibility helpers -------------------------
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)  # optional; slower, can raise on some ops


def main():
    # ------------------------- config block ---------------------------------
    CFG = dict(
        data_root="data/chest_xray",
        img_size=224,
        batch_size=8,
        val_pct=0.10,
        epochs=8,
        lr=1e-4,
        weight_decay=1e-4,
        seed=42,
        aug=True,
        ckpt_best="best_model.pt",
        ckpt_last="last_model.pt",
        num_threads=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(CFG["num_threads"])
    set_all_seeds(CFG["seed"])

    # ------------------------- data -----------------------------------------
    dl_tr, dl_va, dl_te, classes = make_loaders_folder(
        root=CFG["data_root"],
        img_size=CFG["img_size"],
        batch_size=CFG["batch_size"],
        val_pct=CFG["val_pct"],
        seed=CFG["seed"],
        aug=CFG["aug"],
    )
    print("Classes:", classes)

    # ------------------------- model ----------------------------------------
    model = ResNet18Bin().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    # ------------------------- training loop --------------------------------
    best_auroc, best_state = 0.0, None
    for epoch in range(CFG["epochs"]):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {epoch:02d}", leave=False)
        for x, y in pbar:
            x = x.to(device); y = y.float().to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach()))

        # quick val
        auroc, auprc, sens, spec = evaluate(model, dl_va, device)
        print(f"epoch {epoch:02d}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
              f"Sens@0.5={sens:.3f}  Spec@0.5={spec:.3f}")

        # save "last" each epoch (for safety/debug)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, CFG["ckpt_last"])

        # keep best by AUROC
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ------------------------- save best ------------------------------------
    if best_state:
        torch.save(best_state, CFG["ckpt_best"])
        print(f"✅ Saved best model by val AUROC ({best_auroc:.3f}) → {CFG['ckpt_best']}")
        model.load_state_dict(best_state)
        model.to(device)
    else:
        print("⚠️ No best state captured; using last-epoch weights.")

    # ------------------------- threshold tuning -----------------------------
    best_thr = find_best_threshold(model, dl_va, device, mode="youden")
    print(f"Chosen threshold (Youden J): {best_thr:.2f}")

    # ------------------------- test evaluation ------------------------------
    test_auroc, test_auprc, _, _ = evaluate(model, dl_te, device)
    test_ops = evaluate_at_threshold(model, dl_te, thr=best_thr, device=device)
    print(f"TEST  AUROC={test_auroc:.3f}  AUPRC={test_auprc:.3f}  "
          f"Sens@{best_thr:.2f}={test_ops['sensitivity']:.3f}  "
          f"Spec@{best_thr:.2f}={test_ops['specificity']:.3f}  "
          f"Prec={test_ops['precision']:.3f}  Acc={test_ops['accuracy']:.3f}")

    # ------------------------- calibration stage ----------------------------
    print("\n--- Calibration: Temperature Scaling ---")
    logits_va, y_va = logits_labels(model, dl_va, device)
    ts = TemperatureScaler()
    T = ts.fit(logits_va, y_va)   # fit temperature on validation logits
    print(f"Calibrated temperature T = {T:.2f}")

    # optional reliability curves (plots)
    p_va_raw = proba_from_logits(logits_va)
    p_va_cal = proba_from_logits(logits_va / T)
    reliability_curve(y_va, p_va_raw, title="Validation Reliability (raw)")
    reliability_curve(y_va, p_va_cal, title="Validation Reliability (temp-scaled)")

    print("\n✅ Training + Calibration complete. Best model saved at:", CFG["ckpt_best"])


if __name__ == "__main__":
    main()
=======
"""
train.py — robust + user-friendly (final calibrated version)
"""

from __future__ import annotations
import os, random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from .data_folder import make_loaders_folder
from .model import ResNet18Bin
from .eval import evaluate, find_best_threshold, evaluate_at_threshold
from src.metrics import (
    TemperatureScaler, logits_labels, proba_from_logits, reliability_curve
)

# ------------------------- reproducibility helpers -------------------------
def set_all_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)  # optional; slower, can raise on some ops


def main():
    # ------------------------- config block ---------------------------------
    CFG = dict(
        data_root="data/chest_xray",
        img_size=224,
        batch_size=8,
        val_pct=0.10,
        epochs=8,
        lr=1e-4,
        weight_decay=1e-4,
        seed=42,
        aug=True,
        ckpt_best="best_model.pt",
        ckpt_last="last_model.pt",
        num_threads=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(CFG["num_threads"])
    set_all_seeds(CFG["seed"])

    # ------------------------- data -----------------------------------------
    dl_tr, dl_va, dl_te, classes = make_loaders_folder(
        root=CFG["data_root"],
        img_size=CFG["img_size"],
        batch_size=CFG["batch_size"],
        val_pct=CFG["val_pct"],
        seed=CFG["seed"],
        aug=CFG["aug"],
    )
    print("Classes:", classes)

    # ------------------------- model ----------------------------------------
    model = ResNet18Bin().to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    opt = AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])

    # ------------------------- training loop --------------------------------
    best_auroc, best_state = 0.0, None
    for epoch in range(CFG["epochs"]):
        model.train()
        pbar = tqdm(dl_tr, desc=f"epoch {epoch:02d}", leave=False)
        for x, y in pbar:
            x = x.to(device); y = y.float().to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
            pbar.set_postfix(loss=float(loss.detach()))

        # quick val
        auroc, auprc, sens, spec = evaluate(model, dl_va, device)
        print(f"epoch {epoch:02d}  AUROC={auroc:.3f}  AUPRC={auprc:.3f}  "
              f"Sens@0.5={sens:.3f}  Spec@0.5={spec:.3f}")

        # save "last" each epoch (for safety/debug)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, CFG["ckpt_last"])

        # keep best by AUROC
        if auroc > best_auroc:
            best_auroc = auroc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # ------------------------- save best ------------------------------------
    if best_state:
        torch.save(best_state, CFG["ckpt_best"])
        print(f"✅ Saved best model by val AUROC ({best_auroc:.3f}) → {CFG['ckpt_best']}")
        model.load_state_dict(best_state)
        model.to(device)
    else:
        print("⚠️ No best state captured; using last-epoch weights.")

    # ------------------------- threshold tuning -----------------------------
    best_thr = find_best_threshold(model, dl_va, device, mode="youden")
    print(f"Chosen threshold (Youden J): {best_thr:.2f}")

    # ------------------------- test evaluation ------------------------------
    test_auroc, test_auprc, _, _ = evaluate(model, dl_te, device)
    test_ops = evaluate_at_threshold(model, dl_te, thr=best_thr, device=device)
    print(f"TEST  AUROC={test_auroc:.3f}  AUPRC={test_auprc:.3f}  "
          f"Sens@{best_thr:.2f}={test_ops['sensitivity']:.3f}  "
          f"Spec@{best_thr:.2f}={test_ops['specificity']:.3f}  "
          f"Prec={test_ops['precision']:.3f}  Acc={test_ops['accuracy']:.3f}")

    # ------------------------- calibration stage ----------------------------
    print("\n--- Calibration: Temperature Scaling ---")
    logits_va, y_va = logits_labels(model, dl_va, device)
    ts = TemperatureScaler()
    T = ts.fit(logits_va, y_va)   # fit temperature on validation logits
    print(f"Calibrated temperature T = {T:.2f}")

    # optional reliability curves (plots)
    p_va_raw = proba_from_logits(logits_va)
    p_va_cal = proba_from_logits(logits_va / T)
    reliability_curve(y_va, p_va_raw, title="Validation Reliability (raw)")
    reliability_curve(y_va, p_va_cal, title="Validation Reliability (temp-scaled)")

    print("\n✅ Training + Calibration complete. Best model saved at:", CFG["ckpt_best"])


if __name__ == "__main__":
    main()
>>>>>>> 797ec99 (Clean repo and apply final .gitignore)
