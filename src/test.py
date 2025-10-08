<<<<<<< HEAD
"""
test.py — evaluate best_model.pt with optional threshold control.
"""

import argparse
import torch
from .model import ResNet18Bin
from .data_folder import make_loaders_folder
from .metrics import (logits_labels, proba_from_logits, print_report,
                      pick_screening, pick_rulein)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", type=float, default=None, help="Fixed threshold (e.g., 0.90)")
    parser.add_argument("--policy", type=str, default="youden",
                        choices=["youden", "screening", "rulein"],
                        help="Operating point policy if --thr is not given.")
    parser.add_argument("--min_sens", type=float, default=0.98)
    parser.add_argument("--min_spec", type=float, default=0.90)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    dl_tr, dl_va, dl_te, _ = make_loaders_folder(root="data/chest_xray", img_size=224, batch_size=8, val_pct=0.1)

    # model
    model = ResNet18Bin().to(device)
    best = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(best)
    model.eval()

    # collect logits & probabilities
    logits_va, y_va = logits_labels(model, dl_va, device)
    logits_te, y_te = logits_labels(model, dl_te, device)
    p_va = proba_from_logits(logits_va)
    p_te = proba_from_logits(logits_te)

    # choose threshold
    if args.thr is not None:
        thr = float(args.thr)
        header = f"TEST (Fixed thr {thr:.2f})"
    else:
        if args.policy == "screening":
            thr = pick_screening(y_va, p_va, min_sens=args.min_sens)
            header = f"TEST (Screening; Sens≥{args.min_sens:.2f}; thr={thr:.2f})"
        elif args.policy == "rulein":
            thr = pick_rulein(y_va, p_va, min_spec=args.min_spec)
            header = f"TEST (Rule-in; Spec≥{args.min_spec:.2f}; thr={thr:.2f})"
        else:
            # your train script already prints Youden results; we repeat here:
            from .eval import find_best_threshold
            thr = find_best_threshold_from_probs(y_va, p_va)  # fallback helper below
            header = f"TEST (Youden J; thr={thr:.2f})"

    # final report
    print_report(y_te, p_te, thr, header=header)


# small helper so we don't import torch again for youden on probs
def find_best_threshold_from_probs(y_true, p, grid=None):
    import numpy as np
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_j = 0.5, -1.0
    from sklearn.metrics import confusion_matrix
    for t in grid:
        yb = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yb).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


if __name__ == "__main__":
    main()
=======
"""
test.py — evaluate best_model.pt with optional threshold control.
"""

import argparse
import torch
from .model import ResNet18Bin
from .data_folder import make_loaders_folder
from .metrics import (logits_labels, proba_from_logits, print_report,
                      pick_screening, pick_rulein)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--thr", type=float, default=None, help="Fixed threshold (e.g., 0.90)")
    parser.add_argument("--policy", type=str, default="youden",
                        choices=["youden", "screening", "rulein"],
                        help="Operating point policy if --thr is not given.")
    parser.add_argument("--min_sens", type=float, default=0.98)
    parser.add_argument("--min_spec", type=float, default=0.90)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # data
    dl_tr, dl_va, dl_te, _ = make_loaders_folder(root="data/chest_xray", img_size=224, batch_size=8, val_pct=0.1)

    # model
    model = ResNet18Bin().to(device)
    best = torch.load("best_model.pt", map_location=device)
    model.load_state_dict(best)
    model.eval()

    # collect logits & probabilities
    logits_va, y_va = logits_labels(model, dl_va, device)
    logits_te, y_te = logits_labels(model, dl_te, device)
    p_va = proba_from_logits(logits_va)
    p_te = proba_from_logits(logits_te)

    # choose threshold
    if args.thr is not None:
        thr = float(args.thr)
        header = f"TEST (Fixed thr {thr:.2f})"
    else:
        if args.policy == "screening":
            thr = pick_screening(y_va, p_va, min_sens=args.min_sens)
            header = f"TEST (Screening; Sens≥{args.min_sens:.2f}; thr={thr:.2f})"
        elif args.policy == "rulein":
            thr = pick_rulein(y_va, p_va, min_spec=args.min_spec)
            header = f"TEST (Rule-in; Spec≥{args.min_spec:.2f}; thr={thr:.2f})"
        else:
            # your train script already prints Youden results; we repeat here:
            from .eval import find_best_threshold
            thr = find_best_threshold_from_probs(y_va, p_va)  # fallback helper below
            header = f"TEST (Youden J; thr={thr:.2f})"

    # final report
    print_report(y_te, p_te, thr, header=header)


# small helper so we don't import torch again for youden on probs
def find_best_threshold_from_probs(y_true, p, grid=None):
    import numpy as np
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    best_t, best_j = 0.5, -1.0
    from sklearn.metrics import confusion_matrix
    for t in grid:
        yb = (p >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, yb).ravel()
        sens = tp / (tp + fn + 1e-9)
        spec = tn / (tn + fp + 1e-9)
        j = sens + spec - 1.0
        if j > best_j:
            best_j, best_t = j, float(t)
    return best_t


if __name__ == "__main__":
    main()
>>>>>>> 797ec99 (Clean repo and apply final .gitignore)
