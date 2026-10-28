"""ResNet classification baseline via torchvision / timm."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_resnet(
    encoder_name: str = "resnet50",
    num_classes: int = 2,
    in_channels: int = 3,
    pretrained: bool = True,
    dropout: float = 0.0,
) -> nn.Module:
    """Build a ResNet classifier; adapts first conv for non-RGB medical inputs."""
    try:
        import torchvision.models as tvm
    except ImportError as exc:
        raise ImportError("torchvision is required for ResNet baseline") from exc

    weights = "DEFAULT" if pretrained else None
    builders = {
        "resnet18": tvm.resnet18,
        "resnet34": tvm.resnet34,
        "resnet50": tvm.resnet50,
        "resnet101": tvm.resnet101,
    }
    if encoder_name not in builders:
        raise ValueError(f"Unsupported ResNet: {encoder_name}. Choose from {list(builders)}")

    model = builders[encoder_name](weights=weights)

    if in_channels != 3:
        old = model.conv1
        model.conv1 = nn.Conv2d(
            in_channels,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
            bias=False,
        )
        if pretrained and in_channels == 1:
            with torch.no_grad():
                model.conv1.weight.copy_(old.weight.mean(dim=1, keepdim=True))

    in_features = model.fc.in_features
    head: list[nn.Module] = []
    if dropout > 0:
        head.append(nn.Dropout(dropout))
    head.append(nn.Linear(in_features, num_classes))
    model.fc = nn.Sequential(*head)
    return model
