#!/usr/bin/env python3
"""Download Medical Segmentation Decathlon (MSD) tasks via MONAI.

Example:
    python data/download_scripts/download_msd.py --task Task01_BrainTumour --out data/raw/msd
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download MSD dataset via MONAI")
    parser.add_argument(
        "--task",
        type=str,
        default="Task01_BrainTumour",
        help="MSD task name, e.g. Task01_BrainTumour",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="data/raw/msd",
        help="Output directory for downloaded data",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from monai.apps import DecathlonDataset
    except ImportError as exc:
        raise SystemExit(
            "MONAI is required. Install with: pip install monai"
        ) from exc

    print(f"[MSD] downloading task={args.task} -> {out_dir.resolve()}")
    # DecathlonDataset will download and extract if missing
    _ = DecathlonDataset(
        root_dir=str(out_dir),
        task=args.task,
        section="training",
        download=True,
        progress=True,
    )
    print("[MSD] done.")


if __name__ == "__main__":
    main()
