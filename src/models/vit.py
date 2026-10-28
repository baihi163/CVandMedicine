"""Vision Transformer baseline via timm."""

from __future__ import annotations

import torch.nn as nn


def build_vit(
    num_classes: int = 2,
    in_channels: int = 3,
    image_size: int = 224,
    pretrained: bool = True,
    dropout: float = 0.0,
    model_name: str = "vit_base_patch16_224",
) -> nn.Module:
    """Build a ViT classifier suitable for 2D medical / face-medical images."""
    try:
        import timm
    except ImportError as exc:
        raise ImportError("timm is required for ViT baseline") from exc

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_channels,
        img_size=image_size,
        drop_rate=dropout,
    )
    return model
