#!/usr/bin/env python3
"""Unified training entry (argparse + PyTorch Lightning style).

Examples:
    python -m src.train --config configs/default.yaml
    python src/train.py --config configs/seg_unet.yaml --max_epochs 50
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# allow `python src/train.py` from repo root
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.config import load_config, save_config
from src.utils.metrics import compute_segmentation_metrics, logits_to_pred


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical CV training")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="cuda | cpu | mps")
    parser.add_argument("--fast_dev_run", action="store_true", help="1-batch smoke test")
    return parser.parse_args()


def deep_update(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for k, v in overrides.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        elif v is not None:
            base[k] = v
    return base


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "dice":
        return DiceLoss()
    if name == "dice_ce":
        return DiceCELoss()
    raise ValueError(f"Unknown loss: {name}")


class DiceLoss(nn.Module):
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] > 1:
            probs = torch.softmax(logits, dim=1)
            if target.ndim == logits.ndim:
                target = target.argmax(dim=1)
            target_oh = F.one_hot(target.long(), num_classes=logits.shape[1]).permute(0, -1, *range(1, target.ndim)).float()
        else:
            probs = torch.sigmoid(logits)
            target_oh = target.float()
            if target_oh.ndim == probs.ndim - 1:
                target_oh = target_oh.unsqueeze(1)

        dims = tuple(range(2, probs.ndim))
        inter = torch.sum(probs * target_oh, dim=dims)
        denom = torch.sum(probs + target_oh, dim=dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        return 1 - dice.mean()


class DiceCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == logits.ndim:
            target_ce = target.argmax(dim=1)
        else:
            target_ce = target.long()
            if target_ce.ndim == logits.ndim - 1 and target_ce.shape[1] == 1:
                target_ce = target_ce.squeeze(1)
        return self.dice(logits, target) + self.ce(logits, target_ce)


class MedicalLitModule:
    """Lightweight Lightning-style training module (works without Lightning)."""

    def __init__(self, cfg: dict[str, Any], model: nn.Module, device: torch.device):
        self.cfg = cfg
        self.model = model.to(device)
        self.device = device
        self.criterion = build_loss(str(cfg["train"].get("loss", "dice_ce")))
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        self.use_amp = bool(cfg["train"].get("amp", True)) and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _build_optimizer(self):
        lr = float(self.cfg["train"]["lr"])
        wd = float(self.cfg["train"].get("weight_decay", 0.0))
        name = str(self.cfg["train"].get("optimizer", "adamw")).lower()
        params = self.model.parameters()
        if name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=wd)
        if name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=0.9)
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    def _build_scheduler(self):
        name = str(self.cfg["train"].get("scheduler", "cosine")).lower()
        if name == "none":
            return None
        epochs = int(self.cfg["train"]["max_epochs"])
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=max(epochs // 3, 1), gamma=0.1)
        return torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

    def _batch_target(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        # segmentation prefers mask; classification uses label
        model_name = str(self.cfg["model"].get("name", "unet")).lower()
        if model_name == "unet":
            target = batch["mask"].to(self.device)
            if target.ndim == 4 and target.shape[1] == 1:
                target = target.squeeze(1).long()
            elif target.ndim == 5 and target.shape[1] == 1:
                target = target.squeeze(1).long()
            else:
                target = target.long()
            return target
        return batch["label"].to(self.device).long()

    def training_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.train()
        images = batch["image"].to(self.device)
        target = self._batch_target(batch)

        self.optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            logits = self.model(images)
            # classification heads return (B,C); ensure CE target matches
            if logits.ndim == 2 and target.ndim > 1:
                target = target.view(target.shape[0], -1)[:, 0]
            loss = self.criterion(logits, target)

        self.scaler.scale(loss).backward()
        clip = self.cfg["train"].get("gradient_clip_val")
        if clip:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(clip))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return {"loss": float(loss.detach().cpu())}

    @torch.no_grad()
    def validation_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        self.model.eval()
        images = batch["image"].to(self.device)
        target = self._batch_target(batch)
        logits = self.model(images)
        if logits.ndim == 2 and target.ndim > 1:
            target = target.view(target.shape[0], -1)[:, 0]
        loss = self.criterion(logits, target)

        metrics: dict[str, float] = {"loss": float(loss.cpu())}
        model_name = str(self.cfg["model"].get("name", "unet")).lower()
        if model_name == "unet":
            pred = logits_to_pred(logits)
            # average over batch
            dices, ious = [], []
            for i in range(pred.shape[0]):
                m = compute_segmentation_metrics(pred[i], target[i])
                dices.append(m["dice"])
                ious.append(m["iou"])
            metrics["dice"] = float(sum(dices) / max(len(dices), 1))
            metrics["iou"] = float(sum(ious) / max(len(ious), 1))
        else:
            pred = logits.argmax(dim=1)
            metrics["acc"] = float((pred == target).float().mean().cpu())
        return metrics


def _resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train(cfg: dict[str, Any], fast_dev_run: bool = False, device_name: str | None = None) -> Path:
    from rich.console import Console
    from rich.progress import track

    console = Console()
    set_seed(int(cfg["experiment"].get("seed", 42)))
    device = _resolve_device(device_name)

    out_dir = Path(cfg["experiment"].get("output_dir", "outputs")) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    train_loader, val_loader, _ = build_dataloaders(cfg)
    model = build_model(cfg)
    lit = MedicalLitModule(cfg, model, device)

    console.print(f"[bold green]Device[/]: {device}")
    console.print(f"[bold green]Model[/]: {cfg['model']['name']} | samples train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    best_score = -float("inf") if cfg["train"].get("monitor_mode", "max") == "max" else float("inf")
    monitor = str(cfg["train"].get("monitor", "val/dice")).split("/")[-1]
    mode = str(cfg["train"].get("monitor_mode", "max"))
    patience = int(cfg["train"].get("early_stopping_patience", 20))
    wait = 0
    max_epochs = int(cfg["train"]["max_epochs"])
    if fast_dev_run:
        max_epochs = 1

    for epoch in range(1, max_epochs + 1):
        train_losses = []
        iterable = train_loader
        if fast_dev_run:
            iterable = [next(iter(train_loader))]
        for batch in track(iterable, description=f"Epoch {epoch}/{max_epochs} train", leave=False):
            logs = lit.training_step(batch)
            train_losses.append(logs["loss"])

        val_logs: list[dict[str, float]] = []
        val_iter: DataLoader | list = val_loader
        if fast_dev_run:
            val_iter = [next(iter(val_loader))]
        for batch in val_iter:
            val_logs.append(lit.validation_step(batch))

        avg_train = sum(train_losses) / max(len(train_losses), 1)
        avg_val = {k: sum(d[k] for d in val_logs) / max(len(val_logs), 1) for k in val_logs[0]}
        if lit.scheduler is not None:
            lit.scheduler.step()

        score = avg_val.get(monitor, avg_val.get("loss", 0.0))
        console.print(
            f"Epoch {epoch:03d} | train_loss={avg_train:.4f} | "
            + " | ".join(f"val_{k}={v:.4f}" for k, v in avg_val.items())
        )

        improved = score > best_score if mode == "max" else score < best_score
        if improved:
            best_score = score
            wait = 0
            ckpt_path = ckpt_dir / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": lit.model.state_dict(),
                    "optimizer_state": lit.optimizer.state_dict(),
                    "config": cfg,
                    "best_score": best_score,
                },
                ckpt_path,
            )
            console.print(f"  [cyan]saved[/] {ckpt_path} ({monitor}={best_score:.4f})")
        else:
            wait += 1
            if wait >= patience:
                console.print(f"[yellow]Early stopping[/] at epoch {epoch}")
                break

        if epoch % 10 == 0:
            torch.save({"epoch": epoch, "model_state": lit.model.state_dict(), "config": cfg}, ckpt_dir / f"epoch_{epoch:03d}.pt")

    console.print(f"[bold]Done.[/] Best {monitor}={best_score:.4f} → {ckpt_dir / 'best.pt'}")
    return ckpt_dir / "best.pt"


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    overrides = {
        "experiment": {
            "output_dir": args.output_dir,
            "seed": args.seed,
        },
        "data": {"batch_size": args.batch_size},
        "train": {"max_epochs": args.max_epochs, "lr": args.lr},
    }
    deep_update(cfg, overrides)
    train(cfg, fast_dev_run=args.fast_dev_run, device_name=args.device)


if __name__ == "__main__":
    main()
