"""Face / craniofacial medical imaging dataset template.

Typical use-cases: facial palsy grading, craniofacial deformity screening,
dermatology-on-face, surgical outcome scoring.

Expected layout:
  root/
    images/*.jpg|png
    labels.csv          # columns: filename,label
  or
  root/
    class_name_a/*.jpg
    class_name_b/*.jpg
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch.utils.data import Dataset


class FaceMedicalDataset(Dataset):
    """2D RGB-oriented loader for face-related medical / clinical photos."""

    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        image_size: int = 224,
        is_3d: bool = False,  # kept for API parity; face data is 2D
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        transform: Callable | None = None,
    ):
        if is_3d:
            raise ValueError("FaceMedicalDataset is 2D-only")

        self.root = Path(root)
        self.split = split
        self.image_size = image_size
        self.transform = transform
        self.samples = self._discover()
        self.samples = self._split(self.samples, train_ratio, val_ratio, split)
        self._dummy = len(self.samples) == 0

    def _discover(self) -> list[dict]:
        samples: list[dict] = []
        labels_csv = self.root / "labels.csv"
        images_dir = self.root / "images"

        if labels_csv.exists() and images_dir.exists():
            with labels_csv.open("r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    fname = row.get("filename") or row.get("file") or row.get("image")
                    label = row.get("label") or row.get("class")
                    if not fname or label is None:
                        continue
                    path = images_dir / fname
                    if path.exists():
                        samples.append({"image": path, "label": int(label)})
            return samples

        class_dirs = [d for d in sorted(self.root.iterdir()) if d.is_dir()]
        for label, cdir in enumerate(class_dirs):
            for p in sorted(cdir.glob("*")):
                if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
                    samples.append({"image": p, "label": label})
        return samples

    def _split(self, samples: list[dict], train_ratio: float, val_ratio: float, split: str) -> list[dict]:
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

    def __len__(self) -> int:
        return max(len(self.samples), 1 if self._dummy else 0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self._dummy:
            image = torch.zeros(3, self.image_size, self.image_size, dtype=torch.float32)
            return {"image": image, "label": torch.tensor(0, dtype=torch.long), "mask": torch.zeros(1, self.image_size, self.image_size)}

        item = self.samples[index]
        image = self._load_rgb(item["image"])
        if self.transform is not None:
            # albumentations-style: transform(image=np) -> dict
            arr = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            out = self.transform(image=arr)
            image = torch.from_numpy(out["image"]).float()
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            if image.max() > 1:
                image = image / 255.0

        return {
            "image": image.float(),
            "label": torch.tensor(int(item["label"]), dtype=torch.long),
            "mask": torch.zeros(1, self.image_size, self.image_size),
        }

    def _load_rgb(self, path: Path) -> torch.Tensor:
        try:
            import cv2

            bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise FileNotFoundError(path)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            return tensor
        except Exception:
            from PIL import Image

            img = Image.open(path).convert("RGB").resize((self.image_size, self.image_size))
            arr = np.asarray(img, dtype=np.float32) / 255.0
            return torch.from_numpy(arr).permute(2, 0, 1)
