"""Visualization helpers for medical CV debugging / paper figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _to_hwc_uint8(image: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]
    if arr.ndim == 3 and arr.shape[0] > 3 and arr.shape[-1] not in (1, 3):
        # maybe (D,H,W) volume -> take mid-slice
        arr = arr[arr.shape[0] // 2]
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    return arr


def overlay_mask(
    image: torch.Tensor | np.ndarray,
    mask: torch.Tensor | np.ndarray,
    color: tuple[int, int, int] = (255, 64, 64),
    alpha: float = 0.45,
) -> np.ndarray:
    """Overlay a binary mask on an image; returns HxWx3 uint8."""
    base = _to_hwc_uint8(image).astype(np.float32)
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().numpy()
    m = np.asarray(mask).squeeze()
    if m.ndim == 3:
        m = m[m.shape[0] // 2]
    m = (m > 0.5).astype(np.float32)
    color_arr = np.array(color, dtype=np.float32).reshape(1, 1, 3)
    out = base.copy()
    out[m > 0] = (1 - alpha) * out[m > 0] + alpha * color_arr
    return np.clip(out, 0, 255).astype(np.uint8)


def save_prediction_grid(
    images: torch.Tensor,
    masks: torch.Tensor | None,
    preds: torch.Tensor,
    out_path: str | Path,
    max_items: int = 4,
) -> Path:
    """Save a simple image | GT | Pred grid with matplotlib."""
    import matplotlib.pyplot as plt

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = min(images.shape[0], max_items)
    cols = 3 if masks is not None else 2
    fig, axes = plt.subplots(n, cols, figsize=(3 * cols, 3 * n))
    if n == 1:
        axes = np.expand_dims(axes, 0)

    for i in range(n):
        img = _to_hwc_uint8(images[i])
        pred_i = preds[i]
        if isinstance(pred_i, torch.Tensor) and pred_i.ndim > 2:
            pred_i = pred_i.argmax(dim=0) if pred_i.shape[0] > 1 else pred_i.squeeze(0)

        axes[i, 0].imshow(img)
        axes[i, 0].set_title("Image")
        axes[i, 0].axis("off")

        col = 1
        if masks is not None:
            gt = overlay_mask(images[i], masks[i])
            axes[i, col].imshow(gt)
            axes[i, col].set_title("GT")
            axes[i, col].axis("off")
            col += 1

        pr = overlay_mask(images[i], pred_i, color=(64, 160, 255))
        axes[i, col].imshow(pr)
        axes[i, col].set_title("Pred")
        axes[i, col].axis("off")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
