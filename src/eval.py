#!/usr/bin/env python3
"""Evaluation script for Medical CV checkpoints.

Examples:
    python -m src.eval --config configs/default.yaml --checkpoint outputs/.../best.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.datasets import build_dataloaders
from src.models import build_model
from src.utils.config import load_config
from src.utils.metrics import compute_segmentation_metrics, logits_to_pred, sensitivity, specificity
from src.utils.visualization import save_prediction_grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical CV evaluation")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(cfg: dict, checkpoint: str, split: str, device_name: str | None, output_dir: str | None) -> dict:
    from rich.console import Console
    from rich.table import Table

    console = Console()
    device = _device(device_name)
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    loader = val_loader if split == "val" else test_loader

    model = build_model(cfg)
    ckpt = torch.load(checkpoint, map_location=device)
    state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    out_dir = Path(output_dir or Path(checkpoint).parent.parent / "eval")
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_dir = out_dir / "predictions"
    pred_dir.mkdir(exist_ok=True)

    model_name = str(cfg["model"].get("name", "unet")).lower()
    metric_names = cfg.get("eval", {}).get("metrics", ["dice", "iou", "hd95", "sensitivity", "specificity"])
    accum = {m: [] for m in metric_names}
    save_preds = bool(cfg.get("eval", {}).get("save_predictions", True))

    for step, batch in enumerate(loader):
        images = batch["image"].to(device)
        logits = model(images)

        if model_name == "unet":
            target = batch["mask"].to(device)
            if target.ndim >= 4 and target.shape[1] == 1:
                target = target.squeeze(1)
            pred = logits_to_pred(logits)
            for i in range(pred.shape[0]):
                m = compute_segmentation_metrics(pred[i], target[i])
                for k in accum:
                    if k in m:
                        accum[k].append(m[k])
            if save_preds and step == 0:
                save_prediction_grid(images.cpu(), target.cpu(), pred.cpu(), pred_dir / "grid.png")
        else:
            labels = batch["label"].to(device)
            pred = logits.argmax(dim=1)
            correct = (pred == labels).float()
            if "accuracy" in accum:
                accum["accuracy"].extend(correct.cpu().tolist())
            # binary sens/spec if 2-class
            if int(cfg["data"].get("num_classes", 2)) == 2:
                if "sensitivity" in accum:
                    accum["sensitivity"].append(sensitivity(pred, labels))
                if "specificity" in accum:
                    accum["specificity"].append(specificity(pred, labels))

    summary = {k: float(sum(v) / len(v)) if v else float("nan") for k, v in accum.items()}
    # filter nan-heavy hd95 for empty masks
    for k, v in list(summary.items()):
        if v != v:  # NaN
            vals = [x for x in accum[k] if x == x]
            summary[k] = float(sum(vals) / len(vals)) if vals else float("nan")

    table = Table(title=f"Eval ({split})")
    table.add_column("Metric")
    table.add_column("Value")
    for k, v in summary.items():
        table.add_row(k, f"{v:.4f}")
    console.print(table)

    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    console.print(f"Saved metrics → {out_dir / 'metrics.json'}")
    return summary


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if cfg.get("eval") is not None:
        cfg["eval"]["checkpoint"] = args.checkpoint
    evaluate(cfg, args.checkpoint, args.split, args.device, args.output_dir)


if __name__ == "__main__":
    main()
