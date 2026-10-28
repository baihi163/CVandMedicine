# Data

Place medical imaging datasets here. Large files are ignored by Git.

Suggested layout:

```text
data/
├── raw/              # original downloads
├── processed/        # preprocessed / cached tensors
├── download_scripts/ # public dataset fetchers
└── README.md
```

## Quick start

```bash
# example: Medical Segmentation Decathlon Task01 BrainTumour
python data/download_scripts/download_msd.py --task Task01_BrainTumour --out data/raw/msd
```

Do **not** commit PHI / clinical patient data.
