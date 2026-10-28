#!/usr/bin/env python3
"""Placeholder script for ISIC skin-lesion datasets.

ISIC releases often require registration / accepting terms.
This script documents the expected directory layout and verifies a local path.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare / verify ISIC dataset folder")
    parser.add_argument(
        "--root",
        type=str,
        default="data/raw/isic",
        help="Local ISIC root directory",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    images = root / "images"
    masks = root / "masks"
    images.mkdir(parents=True, exist_ok=True)
    masks.mkdir(parents=True, exist_ok=True)

    n_img = len(list(images.glob("*")))
    n_msk = len(list(masks.glob("*")))
    print(f"[ISIC] root={root.resolve()}")
    print(f"[ISIC] images={n_img}, masks={n_msk}")
    print(
        "[ISIC] Download manually from https://www.isic-archive.com/ "
        "and place files under images/ and masks/."
    )


if __name__ == "__main__":
    main()
