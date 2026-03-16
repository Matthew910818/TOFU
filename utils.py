import os
import random
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Optional

# -----------------------------
# Basic settings and file operations
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def to_str_array(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype.kind in ("S", "a"):
        return np.char.decode(a, "utf-8", errors="ignore")
    return a.astype(str)

def list_npz_files(data_dir: str) -> List[str]:
    files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith(".npz")]
    files.sort()
    return files

def split_trials(files: List[str], val_ratio: float, seed: int):
    rnd = random.Random(seed)
    files2 = files[:]
    rnd.shuffle(files2)
    n_val = max(1, int(len(files2) * val_ratio))
    val_files = files2[:n_val]
    train_files = files2[n_val:]
    return train_files, val_files

# -----------------------------
# Array reshaping and masking
# -----------------------------
def reshape_ev_to_grid(ev: np.ndarray, H=9, W=7):
    return ev.reshape(ev.shape[0], H, W)

def reshape_dxdy_to_grid(dxdy: np.ndarray, H=9, W=7):
    return dxdy.reshape(dxdy.shape[0], H, W, 2)

def make_mask(L: int, H: int, W: int, mask_ratio: float):
    """
    Returns mask of shape (L, H, W) with True on masked positions.
    (Used in pretrain.py)
    """
    total = L * H * W
    k = max(1, int(total * mask_ratio))
    idx = torch.randperm(total)[:k]
    mask = torch.zeros(total, dtype=torch.bool)
    mask[idx] = True
    return mask.view(L, H, W)

# -----------------------------
# Statistics and signal processing
# -----------------------------
def mad_stats(x: np.ndarray, eps: float = 1e-8):
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + eps
    sigma = 1.4826 * mad
    return med, sigma

def robust_zscore(x: np.ndarray, eps: float = 1e-8):
    med, sig = mad_stats(x, eps=eps)
    return (x - med) / max(sig, eps)

def moving_average_1d(x: np.ndarray, k: int):
    if k <= 1 or x.size == 0:
        return x.astype(np.float32, copy=False)
    k = min(k, x.size)
    pad_l = k // 2
    pad_r = k - 1 - pad_l
    xp = np.pad(x.astype(np.float32), (pad_l, pad_r), mode="edge")
    ker = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(xp, ker, mode="valid")

def hist_entropy_1d(v: np.ndarray, bins: int, vmin: float, vmax: float):
    if vmax <= vmin:
        return 0.0
    h, _ = np.histogram(v, bins=bins, range=(vmin, vmax), density=False)
    s = h.sum()
    if s <= 0:
        return 0.0
    p = h.astype(np.float64) / float(s)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

def hist_entropy_2d(x: np.ndarray, y: np.ndarray, bins: int, xmin: float, xmax: float, ymin: float, ymax: float):
    if xmax <= xmin or ymax <= ymin:
        return 0.0
    h2, _, _ = np.histogram2d(x, y, bins=bins, range=[[xmin, xmax], [ymin, ymax]], density=False)
    s = h2.sum()
    if s <= 0:
        return 0.0
    p = (h2.astype(np.float64) / float(s)).ravel()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))

# -----------------------------
# Losses and Metrics (for Finetuning)
# -----------------------------
def ordinal_targets(y: torch.Tensor, num_levels: int = 3) -> torch.Tensor:
    cuts = []
    for k in range(num_levels - 1):
        cuts.append((y > k).float())
    return torch.stack(cuts, dim=1)

def ordinal_bce_loss(logits: torch.Tensor, y: torch.Tensor, pos_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
    tgt = ordinal_targets(y, num_levels=3).to(logits.device)
    if pos_weight is not None:
        pos_weight = pos_weight.to(logits.device)
    return F.binary_cross_entropy_with_logits(logits, tgt, pos_weight=pos_weight)

def ordinal_decode(logits: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    return (p > 0.5).sum(dim=1).long()

def confusion_matrix_3(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((3, 3), dtype=np.int64)
    for t, p in zip(y_true.tolist(), y_pred.tolist()):
        cm[int(t), int(p)] += 1
    return cm

def augment_tactile_batch(x: torch.Tensor, p: float = 0.5) -> torch.Tensor:
    x_aug = x.clone()
    B = x_aug.shape[0]
    
    flip_h = torch.rand(B, device=x.device) < p
    if flip_h.any():
        x_aug[flip_h] = torch.flip(x_aug[flip_h], dims=[-1])
        x_aug[flip_h, :, 1, :, :] *= -1.0

    flip_v = torch.rand(B, device=x.device) < p
    if flip_v.any():
        x_aug[flip_v] = torch.flip(x_aug[flip_v], dims=[-2])
        x_aug[flip_v, :, 2, :, :] *= -1.0
    return x_aug