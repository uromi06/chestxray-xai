"""
data_folder.py
--------------
This file prepares the dataset and DataLoaders for training, validation, and testing.
We use torchvision.datasets.ImageFolder since the dataset is already structured
like: train/NORMAL, train/PNEUMONIA, test/NORMAL, test/PNEUMONIA, etc.

Author: Urmi Bhattacharyya
Date: September 2025

NOTE:
The original Kaggle dataset includes a small `val/` folder (16 images).
Instead of using that, we create our own validation split (10% of the training data)
for more reliable evaluation.
"""

from collections import defaultdict
from math import floor
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# --- helper: convert grayscale (1 ch) -> 3 ch (for ImageNet-pretrained backbones) ---
def to_3ch(tensor_img):
    return tensor_img.repeat(3, 1, 1)

def make_loaders_folder(
    root="data/chest_xray",
    img_size=224,
    batch_size=8,
    val_pct=0.1,
    seed=42,
    aug=True,
):
    """
    Returns:
        dl_tr, dl_va, dl_te: DataLoaders for train/val/test
        classes: ["NORMAL", "PNEUMONIA"]
    """
    # Transforms: light aug on train; eval transforms on val/test
    tf_train = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip() if aug else transforms.Lambda(lambda x: x),
        transforms.RandomRotation(5) if aug else transforms.Lambda(lambda x: x),
        transforms.RandomAffine(
        degrees=5, translate=(0.02, 0.02), scale=(0.98, 1.02)
        ) if aug else transforms.Lambda(lambda x: x),   # <— new mild affine
        transforms.ToTensor(),
        transforms.Lambda(to_3ch),
        transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
    ])
    tf_eval = transforms.Compose([
        transforms.Grayscale(1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Lambda(to_3ch),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Load the full train once with train transforms
    full_train = datasets.ImageFolder(f"{root}/train", transform=tf_train)
    classes = full_train.classes  # ["NORMAL", "PNEUMONIA"]

    # -------- STRATIFIED TRAIN/VAL SPLIT --------
    # targets are 0 for NORMAL, 1 for PNEUMONIA
    targets = full_train.targets

    # collect indices per class
    cls_idx = defaultdict(list)
    for i, t in enumerate(targets):
        cls_idx[int(t)].append(i)

    rng = np.random.default_rng(seed)
    val_idx, train_idx = [], []
    for _, idxs in cls_idx.items():
        idxs = np.array(idxs)
        rng.shuffle(idxs)
        n_val_c = floor(len(idxs) * val_pct)
        val_idx.extend(idxs[:n_val_c])
        train_idx.extend(idxs[n_val_c:])

    val_idx = np.array(val_idx)
    train_idx = np.array(train_idx)

    # Build the actual Subset datasets
    ds_tr = Subset(full_train, train_idx)

    # for validation we reload the same files but with eval transforms
    full_train_eval = datasets.ImageFolder(f"{root}/train", transform=tf_eval)
    ds_va = Subset(full_train_eval, val_idx)

    # Test set uses eval transforms
    ds_te = datasets.ImageFolder(f"{root}/test", transform=tf_eval)

    # Windows/CPU-friendly loader settings
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True,
                       num_workers=0, pin_memory=False)
    dl_va = DataLoader(ds_va, batch_size=batch_size * 2, shuffle=False,
                       num_workers=0, pin_memory=False)
    dl_te = DataLoader(ds_te, batch_size=batch_size * 2, shuffle=False,
                       num_workers=0, pin_memory=False)

    return dl_tr, dl_va, dl_te, classes