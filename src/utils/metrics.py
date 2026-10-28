"""Medical segmentation / classification metrics.

Implements: Dice, IoU, HD95, Sensitivity, Specificity.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def _to_numpy_binary(x: torch.Tensor | np.ndarray, threshold: float = 0.5) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    if x.dtype != bool:
        x = x > threshold
    return x.astype(bool)


def dice_score(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Soft Dice on binary masks (flattened)."""
    p = _to_numpy_binary(pred, threshold).reshape(-1)
    t = _to_numpy_binary(target, threshold).reshape(-1)
    inter = np.logical_and(p, t).sum()
    return float((2.0 * inter + eps) / (p.sum() + t.sum() + eps))


def iou_score(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Intersection over Union (Jaccard)."""
    p = _to_numpy_binary(pred, threshold).reshape(-1)
    t = _to_numpy_binary(target, threshold).reshape(-1)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float((inter + eps) / (union + eps))


def sensitivity(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """Recall / True Positive Rate."""
    p = _to_numpy_binary(pred, threshold).reshape(-1)
    t = _to_numpy_binary(target, threshold).reshape(-1)
    tp = np.logical_and(p, t).sum()
    fn = np.logical_and(~p, t).sum()
    return float((tp + eps) / (tp + fn + eps))


def specificity(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-7,
) -> float:
    """True Negative Rate."""
    p = _to_numpy_binary(pred, threshold).reshape(-1)
    t = _to_numpy_binary(target, threshold).reshape(-1)
    tn = np.logical_and(~p, ~t).sum()
    fp = np.logical_and(p, ~t).sum()
    return float((tn + eps) / (tn + fp + eps))


def hausdorff_distance_95(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
    spacing: tuple[float, ...] | None = None,
) -> float:
    """95th percentile Hausdorff distance (HD95).

    Returns NaN if either mask is empty.
    Prefers MONAI / SciPy; falls back to a surface-distance approximation.
    """
    p = _to_numpy_binary(pred, threshold)
    t = _to_numpy_binary(target, threshold)

    # squeeze channel dims like (1,H,W) / (1,D,H,W)
    p = np.squeeze(p)
    t = np.squeeze(t)

    if p.sum() == 0 or t.sum() == 0:
        return float("nan")

    try:
        from monai.metrics import compute_hausdorff_distance

        # MONAI expects (B,C,...) binary tensors
        pt = torch.from_numpy(p.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        tt = torch.from_numpy(t.astype(np.float32)).unsqueeze(0).unsqueeze(0)
        hd = compute_hausdorff_distance(pt, tt, percentile=95, spacing=spacing)
        return float(hd.item())
    except Exception:
        pass

    try:
        from scipy.ndimage import distance_transform_edt

        def _surface(mask: np.ndarray) -> np.ndarray:
            # boundary ≈ mask XOR eroded(mask); use distance to background as proxy
            return mask ^ _binary_erosion(mask)

        def _binary_erosion(mask: np.ndarray) -> np.ndarray:
            from scipy.ndimage import binary_erosion

            return binary_erosion(mask)

        sp = np.ones(p.ndim) if spacing is None else np.asarray(spacing, dtype=float)
        dt_p = distance_transform_edt(~p, sampling=sp)
        dt_t = distance_transform_edt(~t, sampling=sp)
        surf_p = _surface(p)
        surf_t = _surface(t)
        d_pt = dt_t[surf_p]
        d_tp = dt_p[surf_t]
        if d_pt.size == 0 or d_tp.size == 0:
            return float("nan")
        return float(np.percentile(np.concatenate([d_pt, d_tp]), 95))
    except Exception:
        return float("nan")


def compute_segmentation_metrics(
    pred: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Aggregate standard medical segmentation metrics."""
    # if logits with C>1, take argmax foreground (class >=1) or class-1 channel
    if isinstance(pred, torch.Tensor) and pred.ndim >= 3 and pred.shape[0] > 1:
        pred = pred.argmax(dim=0) if pred.ndim == 3 else pred.argmax(dim=1)

    return {
        "dice": dice_score(pred, target, threshold),
        "iou": iou_score(pred, target, threshold),
        "hd95": hausdorff_distance_95(pred, target, threshold),
        "sensitivity": sensitivity(pred, target, threshold),
        "specificity": specificity(pred, target, threshold),
    }


def logits_to_pred(logits: torch.Tensor) -> torch.Tensor:
    """Convert network logits to discrete prediction map."""
    if logits.ndim >= 2 and logits.shape[1] > 1:
        return logits.argmax(dim=1)
    return (torch.sigmoid(logits) > 0.5).long().squeeze(1)
