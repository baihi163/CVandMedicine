"""Model registry for Medical CV baselines."""

from __future__ import annotations

from typing import Any

from .resnet import build_resnet
from .unet import UNet
from .vit import build_vit


def build_model(cfg: dict[str, Any]):
    """Build a model from a config dict (`cfg['model']` or full config)."""
    model_cfg = cfg.get("model", cfg)
    name = str(model_cfg.get("name", "unet")).lower()
    data_cfg = cfg.get("data", {})

    in_channels = int(data_cfg.get("in_channels", model_cfg.get("in_channels", 1)))
    num_classes = int(data_cfg.get("num_classes", model_cfg.get("num_classes", 2)))
    spatial_dims = 3 if data_cfg.get("is_3d", False) else 2

    if name == "unet":
        features = model_cfg.get("features", (32, 64, 128, 256, 512))
        return UNet(
            in_channels=in_channels,
            out_channels=num_classes,
            features=tuple(features),
            spatial_dims=spatial_dims,
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name == "resnet":
        return build_resnet(
            encoder_name=str(model_cfg.get("encoder_name", "resnet50")),
            num_classes=num_classes,
            in_channels=in_channels,
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
        )
    if name == "vit":
        return build_vit(
            num_classes=num_classes,
            in_channels=in_channels,
            image_size=int(data_cfg.get("image_size", 224)),
            pretrained=bool(model_cfg.get("pretrained", True)),
            dropout=float(model_cfg.get("dropout", 0.0)),
            model_name=str(model_cfg.get("encoder_name", "vit_base_patch16_224")),
        )
    raise ValueError(f"Unknown model name: {name}")


__all__ = ["UNet", "build_resnet", "build_vit", "build_model"]
