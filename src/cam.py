"""
cam.py
------
Grad-CAM visualizations for the Chest X-Ray Pneumonia classifier.

What this script does:
1) Rebuilds val/test loaders (same params as training).
2) Loads 'best_model.pt'.
3) Tunes an operating threshold on the validation set (Youden's J).
4) Scans the test set, finds:
     - up to 3 True Positives (TPs)
     - up to 3 mistakes (False Positives or False Negatives)
5) For each selected image, computes a Grad-CAM heatmap from the last
   conv layer of ResNet18 and saves an overlay to 'demo/'.

   Author: Urmi Bhattacharyya
   Date: September 2025
Run:
    python -m src.cam

Outputs:
    demo/cam_tp_01.jpg, demo/cam_tp_02.jpg, ...
    demo/cam_err_01.jpg, demo/cam_err_02.jpg, ...
"""

from __future__ import annotations
import os
import math
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import cv2  # used for color maps and blending

from .data_folder import make_loaders_folder
from .model import ResNet18Bin
from .eval import (
    evaluate,               # threshold-free metrics
    find_best_threshold,    # tunes threshold on validation
    collect_probs_labels,   # shared helper to get probs/labels
)

# ----------------------- Config (match train.py) -----------------------
DATA_ROOT = "data/chest_xray"
IMG_SIZE = 224
BATCH_SIZE = 8
VAL_PCT = 0.10
SEED = 42
CKPT_PATH = "best_model.pt"
OUT_DIR = "demo"


# ----------------------- Grad-CAM class (generic) ----------------------
class GradCAM:
    """
    Minimal Grad-CAM implementation.

    - Hooks the target conv layer to capture:
        (a) forward activations A (C x H x W)
        (b) backward gradients dY/dA (C x H x W) when we backprop the target score
    - Weights per channel are GAP of gradients.
      CAM = ReLU( sum_k (w_k * A_k) ), then resize to input size.

    Reference:
      Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
      via Gradient-based Localization" (ICCV 2017).
    """
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        # forward hook: save feature maps
        self.fwd_handle = target_layer.register_forward_hook(self._save_activations)
        # backward hook: save gradients wrt feature maps
        self.bwd_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, inp, out):
        # out: [B, C, H, W]
        self.activations = out.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        # grad_output[0]: gradient wrt the module's output  [B, C, H, W]
        self.gradients = grad_output[0].detach()

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def __call__(self, scores: torch.Tensor, upsample_to: Tuple[int, int]) -> np.ndarray:
        """
        Build the CAM for the current forward/backward pass.

        Args:
            scores: tensor of shape [B] (the logit we backpropagated)
            upsample_to: (H, W) of input image for resizing

        Returns:
            cam (numpy array HxW in [0,1])
        """
        # safety checks
        assert self.activations is not None and self.gradients is not None, \
            "Run a forward pass, then call scores.backward() before building CAM."

        # activations/gradients: [B, C, H, W]; we handle batch=1
        A = self.activations      # features
        dYdA = self.gradients     # gradients

        # weights: GAP over spatial dims of gradients: w_k = mean_{i,j} dY/dA_k(i,j)
        weights = dYdA.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]

        # weighted sum over channels
        cam = (weights * A).sum(dim=1, keepdim=True)   # [B, 1, H, W]
        cam = F.relu(cam)                              # ReLU

        # normalize to [0,1]
        cam -= cam.min()
        cam += 1e-9
        cam /= cam.max()

        # upsample to input size and return [H, W]
        cam = F.interpolate(cam, size=upsample_to, mode="bilinear", align_corners=False)
        cam = cam[0, 0].cpu().numpy()
        return cam


# ----------------------- utility: de-normalize for viz ------------------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)

def tensor_to_uint8(img_t: torch.Tensor) -> np.ndarray:
    """
    Inverse of the transforms we used:
      - input is [3, H, W], normalized by ImageNet stats
      - returns uint8 RGB image [H, W, 3] in 0..255
    """
    x = img_t.detach().cpu().numpy()
    x = (x * IMAGENET_STD + IMAGENET_MEAN)  # de-normalize
    x = np.clip(x, 0.0, 1.0)
    x = (x * 255).astype(np.uint8)
    x = np.transpose(x, (1, 2, 0))  # CHW -> HWC
    return x


def overlay_cam(rgb_img: np.ndarray, cam: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    """
    Apply a JET colormap to the CAM and blend with the RGB image.

    Args:
        rgb_img: uint8 RGB [H, W, 3]
        cam: float CAM [H, W] in [0,1]
        alpha: blending weight for heatmap (0..1)

    Returns:
        blended RGB uint8 image
    """
    heat = (cam * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)           # BGR
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)               # → RGB

    blended = (alpha * heat + (1 - alpha) * rgb_img).astype(np.uint8)
    return blended


# ----------------------- choose examples from test ----------------------
def pick_examples(model, dl_va, dl_te, device, n_tp: int = 3, n_err: int = 3):
    """
    1) Tune threshold on validation.
    2) Run test once (no shuffle) to collect probs/labels and indices.
    3) Pick up to n_tp true positives and up to n_err mistakes (FP or FN).

    Returns:
        thr: tuned threshold
        tp_idx: list of dataset indices that are true positives
        err_idx: list of dataset indices that are false positives OR false negatives
    """
    # 1) tune operating point on val
    thr = find_best_threshold(model, dl_va, device, mode="youden")

    # 2) probs/labels for test (no shuffle → dataloader order == dataset order)
    probs, labels = collect_probs_labels(model, dl_te, device)
    preds = (probs >= thr).astype(int)

    # identify TPs and errors
    tp = np.where((preds == 1) & (labels == 1))[0]
    fp = np.where((preds == 1) & (labels == 0))[0]
    fn = np.where((preds == 0) & (labels == 1))[0]

    # pick examples (cap lengths)
    tp_idx = tp[:n_tp].tolist()
    err_idx = (list(fp[:math.ceil(n_err/2)]) + list(fn[:n_err - math.ceil(n_err/2)]))[:n_err]

    return float(thr), tp_idx, err_idx


# ----------------------- main: create and save CAMs ---------------------
def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_num_threads(4)

    # Build loaders (must match train/test scripts)
    dl_tr, dl_va, dl_te, classes = make_loaders_folder(
        root=DATA_ROOT, img_size=IMG_SIZE, batch_size=BATCH_SIZE, val_pct=VAL_PCT, seed=SEED
    )
    print("Classes:", classes)

    # Load model + checkpoint
    model = ResNet18Bin().to(device)
    state = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()

    # Choose the resnet18 last conv layer as Grad-CAM target
    # Our model is wrapped as model.model (see model.py); last block is layer4[-1]
    target_layer = model.model.layer4[-1].conv2
    cam_engine = GradCAM(model, target_layer)

    # Pick examples to visualize
    thr, tp_idx, err_idx = pick_examples(model, dl_va, dl_te, device, n_tp=3, n_err=3)
    print(f"Using tuned threshold (Youden J): {thr:.2f}")
    print(f"Chosen test indices  TPs={tp_idx}  ERR={err_idx}")

    # We’ll access images by index from the underlying ImageFolder
    test_ds = dl_te.dataset  # torchvision.datasets.ImageFolder
    # (dl_te has shuffle=False, so dataloader order == dataset order)

    def make_cam_and_save(idx: int, tag: str, i: int):
        """
        Runs a forward+backward pass for one image to produce a Grad-CAM overlay.
        Saves result as 'demo/cam_<tag>_<i>.jpg'.
        """
        # 1) get tensor + label
        img_t, y = test_ds[idx]                 # transformed tensor [3,H,W], label ∈ {0,1}
        x = img_t.unsqueeze(0).to(device)       # add batch dim

        # 2) forward pass
        out = model(x)                          # [1] logit for Pneumonia
        # 3) zero grads and backprop the *positive* logit
        model.zero_grad(set_to_none=True)
        out.backward(torch.ones_like(out))      # d(logit)/d(features)

        # 4) build CAM using hooked activations+grads
        cam = cam_engine(scores=out, upsample_to=(IMG_SIZE, IMG_SIZE))  # [H,W] in [0,1]

        # 5) get a pretty RGB background (de-normalize)
        rgb = tensor_to_uint8(img_t)            # [H,W,3] uint8
        overlay = overlay_cam(rgb, cam, alpha=0.35)

        # 6) annotate with simple text (threshold, label)
        label_name = classes[int(y)]
        txt = f"{tag.upper()} | true:{label_name} | thr={thr:.2f}"
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        cv2.putText(overlay_bgr, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (255, 255, 255), 2, cv2.LINE_AA)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        # 7) save
        out_path = os.path.join(OUT_DIR, f"cam_{tag}_{i:02d}.jpg")
        Image.fromarray(overlay).save(out_path)
        print(f"saved: {out_path}")

    # Generate and save TPs
    for i, idx in enumerate(tp_idx, start=1):
        make_cam_and_save(idx, tag="tp", i=i)

    # Generate and save errors (FP or FN)
    for i, idx in enumerate(err_idx, start=1):
        make_cam_and_save(idx, tag="err", i=i)

    cam_engine.remove_hooks()
    print("✅ Done. Check the 'demo/' folder for overlays.")


if __name__ == "__main__":
    main()
