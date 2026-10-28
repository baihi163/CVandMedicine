"""Dataset builders for Medical CV."""

from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader

from .face_medical import FaceMedicalDataset
from .generic_medical import GenericMedicalDataset


def build_dataloaders(cfg: dict[str, Any]) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    """Create train / val / (optional) test dataloaders from config."""
    data_cfg = cfg.get("data", cfg)
    name = str(data_cfg.get("name", "generic_medical")).lower()
    root = str(data_cfg.get("root", "data/raw"))
    image_size = int(data_cfg.get("image_size", 256))
    is_3d = bool(data_cfg.get("is_3d", False))
    batch_size = int(data_cfg.get("batch_size", 8))
    num_workers = int(data_cfg.get("num_workers", 4))
    train_ratio = float(data_cfg.get("train_ratio", 0.8))
    val_ratio = float(data_cfg.get("val_ratio", 0.1))

    if name in {"generic_medical", "generic", "medical"}:
        ds_cls = GenericMedicalDataset
    elif name in {"face_medical", "face", "facial"}:
        ds_cls = FaceMedicalDataset
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    train_ds = ds_cls(root=root, split="train", image_size=image_size, is_3d=is_3d, train_ratio=train_ratio, val_ratio=val_ratio)
    val_ds = ds_cls(root=root, split="val", image_size=image_size, is_3d=is_3d, train_ratio=train_ratio, val_ratio=val_ratio)
    test_ds = ds_cls(root=root, split="test", image_size=image_size, is_3d=is_3d, train_ratio=train_ratio, val_ratio=val_ratio)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


__all__ = [
    "GenericMedicalDataset",
    "FaceMedicalDataset",
    "build_dataloaders",
]
