# Medical CV Research Scaffold

医学计算机视觉（Medical Computer Vision）科研项目脚手架：数据管线、基线模型、训练/评测脚本与论文速读模板。

## 目录结构

```text
.
├── configs/                 # YAML 超参数
├── data/
│   ├── download_scripts/    # 公开数据集下载脚本
│   └── ...                  # 大文件已被 .gitignore 忽略
├── docs/paper_notes/        # 论文阅读笔记（含 template.md）
├── src/
│   ├── datasets/            # MONAI / PyTorch 数据加载模板
│   ├── models/              # UNet / ResNet / ViT
│   ├── utils/               # Dice, IoU, HD95, Sens/Spec + 可视化
│   ├── train.py
│   └── eval.py
└── requirements.txt
```

## 快速开始

```bash
# 1) 环境
python -m venv .venv
# Windows:
.venv\Scripts\activate
pip install -r requirements.txt

# 2) 下载示例数据（需网络；MSD 体积较大）
python data/download_scripts/download_msd.py --task Task01_BrainTumour --out data/raw/msd

# 3) 训练（无数据时也会用 dummy batch 跑通脚手架）
python -m src.train --config configs/default.yaml --fast_dev_run

# 4) 评测
python -m src.eval --config configs/default.yaml --checkpoint outputs/medical_cv_baseline/checkpoints/best.pt
```

## 配置说明

| 文件 | 用途 |
|------|------|
| `configs/default.yaml` | 默认分割实验 |
| `configs/seg_unet.yaml` | UNet 分割 |
| `configs/cls_resnet.yaml` | ResNet 分类（含 face_medical） |

常用覆盖：

```bash
python -m src.train --config configs/seg_unet.yaml --max_epochs 50 --batch_size 4 --lr 1e-4
```

## 论文笔记

复制模板后填写：

```bash
copy docs\paper_notes\template.md docs\paper_notes\2024_miccai_example.md
```

模板字段覆盖：临床痛点 → Motivation → Method → Ablation → 对本课题启发。

## 数据约定

- **通用医学**（`generic_medical`）：`data/raw/images` + `data/raw/masks`，或按类别分文件夹。
- **面部医学**（`face_medical`）：`images/` + `labels.csv`，或类别文件夹。

请勿提交含 PHI 的临床数据。
