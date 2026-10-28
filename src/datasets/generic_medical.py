"""Generic medical image dataset template (MONAI + PyTorch).

Expected directory layouts (pick one):

1) Flat image–mask pairs
   root/
     images/*.nii.gz | *.png | *.jpg
     masks/*.nii.gz  | *.png | *.jpg

2) Classification folders
   root/
     class_a/*.png
     class_b/*.png
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".nii", ".gz", ".nrrd", ".mha"}


def _list_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    files = []
    for p in sorted(folder.rglob("*")):
        if not p.is_file():
            continue
        suffix = "".join(p.suffixes).lower() if p.suffixes else p.suffix.lower()
        # handle .nii.gz
        if p.suffix.lower() == ".gz" and p.name.lower().endswith(".nii.gz"):
            files.append(p)
        elif p.suffix.lower() in IMAGE_EXTS and not p.name.lower().endswith(".nii.gz"):
            files.append(p)
        elif suffix.endswith(".nii.gz"):
            files.append(p)
    return files


def _stem_key(path: Path) -> str:
    name = path.name
    if name.lower().endswith(".nii.gz"):
        return name[:-7]
    return path.stem


class GenericMedicalDataset(Dataset):
    """Template loader for general medical imaging (segmentation or classification)."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 256,
        is_3d: bool = False,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        transform: Callable | None = None,
    ):
        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.is_3d = is_3d
        self.transform = transform or self._default_transform()

        self.samples = self._discover_samples()
        self.samples = self._split_samples(self.samples, train_ratio, val_ratio, split)

        if len(self.samples) == 0:
            # keep dataset constructible for scaffolding / dry-runs
            self._dummy = True
        else:
            self._dummy = False

    def _discover_samples(self) -> list[dict]:
        images_dir = self.root / "images"
        masks_dir = self.root / "masks"
        samples: list[dict] = []

        if images_dir.exists():
            images = _list_files(images_dir)
            mask_map = {_stem_key(p): p for p in _list_files(masks_dir)} if masks_dir.exists() else {}
            for img in images:
                key = _stem_key(img)
                samples.append(
                    {
                        "image": img,
                        "mask": mask_map.get(key),
                        "label": None,
                        "task": "seg" if key in mask_map else "cls",
                    }
                )
            return samples

        # classification: one folder per class
        class_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir() and d.name not in {"images", "masks"}]
        for label, cdir in enumerate(class_dirs):
            for img in _list_files(cdir):
                samples.append({"image": img, "mask": None, "label": label, "task": "cls"})
        return samples

    def _split_samples(
        self,
        samples: list[dict],
        train_ratio: float,
        val_ratio: float,
        split: str,
    ) -> list[dict]:
        if not samples:
            return []
        rng = np.random.default_rng(42)
        idx = np.arange(len(samples))
        rng.shuffle(idx)
        n = len(samples)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        if split == "train":
            sel = idx[:n_train]
        elif split == "val":
            sel = idx[n_train : n_train + n_val]
        else:
            sel = idx[n_train + n_val :]
        return [samples[i] for i in sel]

    def _default_transform(self):
        """MONAI transforms when available; otherwise identity-style resize."""
        try:
            from monai.transforms import (
                Compose,
                EnsureChannelFirstd,
                EnsureTyped,
                LoadImaged,
                Resized,
                ScaleIntensityd,
            )

            keys = ["image"]
            # mask handled if present in __getitem__
            spatial_size = (self.image_size,) * (3 if self.is_3d else 2)
            return Compose(
                [
                    LoadImaged(keys=keys),
                    EnsureChannelFirstd(keys=keys),
                    ScaleIntensityd(keys=keys),
                    Resized(keys=keys, spatial_size=spatial_size),
                    EnsureTyped(keys=keys),
                ]
            )
        except ImportError:
            return None

    def __len__(self) -> int:
        return max(len(self.samples), 1 if self._dummy else 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._dummy:
            spatial = (self.image_size,) * (3 if self.is_3d else 2)
            image = torch.zeros(1, *spatial, dtype=torch.float32)
            mask = torch.zeros(1, *spatial, dtype=torch.float32)
            return {"image": image, "mask": mask, "label": torch.tensor(0, dtype=torch.long)}

        item = self.samples[index]
        image_path = str(item["image"])

        if self.transform is not None:
            data = self.transform({"image": image_path})
            image = data["image"]
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(np.asarray(image), dtype=torch.float32)
        else:
            image = self._load_fallback(image_path)

        out: dict[str, torch.Tensor] = {"image": image.float()}

        if item.get("mask") is not None:
            mask = self._load_mask(str(item["mask"]))
            out["mask"] = mask
        else:
            # provide empty mask for API consistency in seg pipelines
            out["mask"] = torch.zeros_like(image[:1])

        if item.get("label") is not None:
            out["label"] = torch.tensor(int(item["label"]), dtype=torch.long)
        else:
            out["label"] = torch.tensor(0, dtype=torch.long)

        return out

    def _load_fallback(self, path: str) -> torch.Tensor:
        try:
            import nibabel as nib

            if path.lower().endswith((".nii", ".nii.gz")):
                arr = nib.load(path).get_fdata(dtype=np.float32)
                tensor = torch.from_numpy(arr)
                if tensor.ndim == 2:
                    tensor = tensor.unsqueeze(0)
                elif tensor.ndim == 3 and not self.is_3d:
                    tensor = tensor.unsqueeze(0)
                return tensor
        except Exception:
            pass

        from PIL import Image

        img = Image.open(path).convert("L" if not path.lower().endswith(("jpg", "jpeg", "png")) else "RGB")
        # default grayscale medical
        if img.mode != "L":
            img = img.convert("L")
        img = img.resize((self.image_size, self.image_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).unsqueeze(0)

    def _load_mask(self, path: str) -> torch.Tensor:
        try:
            from monai.transforms import Compose, EnsureChannelFirstd, EnsureTyped, LoadImaged, Resized

            spatial_size = (self.image_size,) * (3 if self.is_3d else 2)
            tfm = Compose(
                [
                    LoadImaged(keys=["mask"]),
                    EnsureChannelFirstd(keys=["mask"]),
                    Resized(keys=["mask"], spatial_size=spatial_size, mode="nearest"),
                    EnsureTyped(keys=["mask"]),
                ]
            )
            mask = tfm({"mask": path})["mask"]
            if not isinstance(mask, torch.Tensor):
                mask = torch.as_tensor(np.asarray(mask))
            return mask.float()
        except Exception:
            m = self._load_fallback(path)
            return (m > 0).float()
