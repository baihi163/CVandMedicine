#!/usr/bin/env python3
"""Fetch survey PDFs and initialize deep-reading note folders.

Usage (from repo root):
    python scripts/fetch_and_init_papers.py
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

try:
    import requests
except ImportError as exc:
    raise SystemExit("Missing dependency: requests. Run: pip install requests") from exc

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        DownloadColumn,
        Progress,
        TextColumn,
        TimeRemainingColumn,
        TransferSpeedColumn,
    )

    USE_RICH = True
except ImportError:
    USE_RICH = False
    try:
        from tqdm import tqdm
    except ImportError as exc:
        raise SystemExit("Need rich or tqdm. Run: pip install rich tqdm") from exc


ROOT = Path(__file__).resolve().parents[1]
NOTES_DIR = ROOT / "docs" / "paper_notes"
CHUNK_SIZE = 1024 * 64
TIMEOUT = 120

PAPERS = [
    {
        "folder": "01_2023_Nature_GMAI_通用医学大模型",
        "title_zh": "通用医学人工智能（Generalist Medical AI）综述",
        "title_en": "Foundation models for generalist medical artificial intelligence",
        "venue": "Nature · 2023",
        "arxiv_id": "2307.15818",
        "arxiv_abs": "https://arxiv.org/abs/2307.15818",
        "pdf_url": "https://arxiv.org/pdf/2307.15818.pdf",
        "pdf_name": "2307.15818_Nature_GMAI.pdf",
        "note_name": "notes_深度阅读.md",
        "tags": "foundation model / GMAI / multimodal / clinical AI",
        "topic_hint": "通用医学大模型、多模态融合、临床落地路径",
    },
    {
        "folder": "02_2024_ArXiv_MedSeg_医学分割演进_UNet到Mamba",
        "title_zh": "医学图像分割演进综述：从 U-Net 到 Mamba",
        "title_en": "A Survey on Deep Learning for Medical Image Segmentation: From U-Net to Mamba",
        "venue": "arXiv · 2024",
        "arxiv_id": "2407.10986",
        "arxiv_abs": "https://arxiv.org/abs/2407.10986",
        "pdf_url": "https://arxiv.org/pdf/2407.10986.pdf",
        "pdf_name": "2407.10986_MedSeg_UNet_to_Mamba.pdf",
        "note_name": "notes_深度阅读.md",
        "tags": "medical segmentation / U-Net / Transformer / Mamba",
        "topic_hint": "分割骨干演进、长程依赖建模、计算效率与医学场景适配",
    },
    {
        "folder": "03_2021_MedIA_SSL_医学自监督学习综述",
        "title_zh": "医学影像自监督学习综述",
        "title_en": "Self-supervised learning for medical image analysis: a comprehensive review",
        "venue": "Medical Image Analysis (MedIA) · 2021",
        "arxiv_id": "2104.14818",
        "arxiv_abs": "https://arxiv.org/abs/2104.14818",
        "pdf_url": "https://arxiv.org/pdf/2104.14818.pdf",
        "pdf_name": "2104.14818_MedIA_SSL_Survey.pdf",
        "note_name": "notes_深度阅读.md",
        "tags": "self-supervised learning / pretext task / medical imaging",
        "topic_hint": "标注稀缺、预训练范式、向下游分割/诊断迁移",
    },
]


def build_note_markdown(paper: dict) -> str:
    """课题组深度阅读规范：五段结构中文模板。"""
    today = date.today().isoformat()
    return f"""# 深度阅读笔记 · {paper['title_zh']}

> 本笔记遵循课题组深度阅读规范。请对照 PDF 原文逐段填写，避免只抄摘要。

---

## 一、论文基本信息

| 字段 | 内容 |
|------|------|
| **中文题名** | {paper['title_zh']} |
| **英文题名** | {paper['title_en']} |
| **会议 / 期刊 · 年份** | {paper['venue']} |
| **ArXiv** | [{paper['arxiv_id']}]({paper['arxiv_abs']}) |
| **本地 PDF** | `{paper['pdf_name']}` |
| **作者 / 机构** | （阅读时补充） |
| **代码 / 项目页** | （若有） |
| **阅读日期** | {today} |
| **课题标签** | {paper['tags']} |
| **一句话定位** | {paper['topic_hint']} |

---

## 二、行业痛点与已有问题

> 对应规范：临床/科研真实痛点是什么？前人为什么没做好？

### 2.1 核心医学 / 临床痛点

- 临床场景：
- 具体问题（诊断 / 分割 / 预后 / 效率 / 标注成本 …）：
- 对患者或医生工作流的影响：

### 2.2 已有方法的不足（Motivation / Gap）

- 主流路线与代表性工作：
- 关键局限（数据、标注、泛化、可解释性、算力、法规/隐私等）：
- 本文明确切入的研究 gap：

### 2.3 本文要回答的核心问题（1–3 句）

-

---

## 三、核心技术与网络架构

> 对应规范：Method & Architecture —— 讲清「怎么做」，而不只是列名词。

### 3.1 总体方法框架 / Pipeline

```
[在此用文字或 ASCII 画出：输入 → 模块 → 输出]
```

### 3.2 关键网络 / 算法设计

- 骨干或范式（CNN / Transformer / SSM-Mamba / 多模态 LLM …）：
- 关键模块与创新点：
- 损失函数 / 训练策略 / 预训练—微调设定：
- 与先前 SOTA 的本质差异：

### 3.3 综述类特有：技术演进脉络（若适用）

| 阶段 / 年份 | 代表方法 | 解决了什么 | 遗留问题 |
|-------------|----------|------------|----------|
|             |          |            |          |

### 3.4 公式 / 实现细节备忘

-

---

## 四、关键指标 / 实验验证

> 对应规范：主实验是否可信？消融是否支撑核心 claim？

### 4.1 数据与评估协议

| 数据集 | 模态 / 任务 | 划分方式 | 主要指标 |
|--------|-------------|----------|----------|
|        |             |          |          |

### 4.2 主实验结论（最有说服力的 2–3 条）

1.
2.
3.

### 4.3 消融与对照（Ablation）

- 去掉 / 替换哪些组件后指标如何变化？
- 是否存在「只涨一点但故事很大」的指标？

### 4.4 局限与失败案例（作者承认 / 你发现的）

-

---

## 五、对本课题的启发与结合点

> 对应规范：读完必须落到「我们能用什么」。

### 5.1 可直接迁移的技术 / 实验设计

-

### 5.2 与本组方向的结合点（Medical CV / 分割 / 大模型 / SSL …）

-

### 5.3 可复现或可对比的实验清单

- [ ] 复现某张表 / 某条曲线
- [ ] 在本组数据上 soft-reproduce 关键模块
- [ ] 与当前 baseline（UNet / ResNet / ViT）对照

### 5.4 需警惕的坑

- 数据泄漏 / 评估协议不一致 / 实现细节未公开：

### 5.5 下一步行动（本周可执行）

1.
2.
3.

---

## 附录

- 关键引用句（可直接写进 related work）：
- 个人评分（新颖度 / 扎实度 / 可读性，1–5）：_ / _ / _
- 备注：
"""


def download_pdf(url: str, dest: Path, console=None) -> bool:
    """Stream-download a PDF with progress; return True if file looks valid."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    headers = {
        "User-Agent": "MedicalCV-PaperFetcher/1.0 (research; +https://arxiv.org)",
        "Accept": "application/pdf,*/*",
    }

    try:
        with requests.get(url, stream=True, timeout=TIMEOUT, headers=headers, allow_redirects=True) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("content-length") or 0)

            if USE_RICH and console is not None:
                with Progress(
                    TextColumn("[bold blue]{task.description}"),
                    BarColumn(),
                    DownloadColumn(),
                    TransferSpeedColumn(),
                    TimeRemainingColumn(),
                    console=console,
                ) as progress:
                    task_id = progress.add_task(dest.name, total=total or None)
                    with tmp.open("wb") as f:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            if not chunk:
                                continue
                            f.write(chunk)
                            progress.update(task_id, advance=len(chunk))
            else:
                with tmp.open("wb") as f:
                    if total > 0:
                        with tqdm(
                            total=total,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=dest.name,
                        ) as bar:
                            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                bar.update(len(chunk))
                    else:
                        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                            if chunk:
                                f.write(chunk)

        # validate
        size = tmp.stat().st_size
        if size < 10_000:
            tmp.unlink(missing_ok=True)
            return False
        with tmp.open("rb") as f:
            magic = f.read(5)
        if magic != b"%PDF-":
            # arxiv sometimes returns HTML error pages
            tmp.unlink(missing_ok=True)
            return False

        tmp.replace(dest)
        return True
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        if console is not None and USE_RICH:
            console.print(f"[red]Download error:[/] {exc}")
        else:
            print(f"Download error: {exc}", file=sys.stderr)
        return False


def try_export_pdf_from_markdown(md_path: Path, console=None) -> Path | None:
    """Optional: markdown2 + weasyprint. Gracefully degrade if unavailable."""
    pdf_path = md_path.with_suffix(".pdf")
    try:
        import markdown2
    except ImportError:
        _print_vscode_tip(md_path, console)
        return None

    try:
        from weasyprint import HTML
    except Exception:
        # weasyprint often fails without native GTK/Pango deps on Windows
        _print_vscode_tip(md_path, console)
        return None

    try:
        html = markdown2.markdown(
            md_path.read_text(encoding="utf-8"),
            extras=["tables", "fenced-code-blocks", "header-ids"],
        )
        styled = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<style>
body {{ font-family: "Segoe UI", "Microsoft YaHei", sans-serif; line-height: 1.55; margin: 2cm; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #888; padding: 6px 8px; }}
code {{ background: #f4f4f4; padding: 1px 4px; }}
h1, h2, h3 {{ color: #1a1a1a; }}
</style></head><body>{html}</body></html>"""
        HTML(string=styled).write_pdf(str(pdf_path))
        if console is not None and USE_RICH:
            console.print(f"  [green]笔记 PDF 已导出[/]: {pdf_path.name}")
        else:
            print(f"  笔记 PDF 已导出: {pdf_path.name}")
        return pdf_path
    except Exception as exc:
        if console is not None and USE_RICH:
            console.print(f"  [yellow]笔记 PDF 导出失败，保留 Markdown[/]: {exc}")
        else:
            print(f"  笔记 PDF 导出失败，保留 Markdown: {exc}")
        _print_vscode_tip(md_path, console)
        return None


def _print_vscode_tip(md_path: Path, console=None) -> None:
    tip = (
        f"已生成标准 Markdown：{md_path}\n"
        "若需 PDF：在 VS Code 安装扩展 “Markdown PDF”，打开该 md 后 "
        "命令面板执行 “Markdown PDF: Export (pdf)”。"
    )
    if console is not None and USE_RICH:
        console.print(f"  [dim]{tip}[/]")
    else:
        print(f"  {tip}")


def init_paper(paper: dict, console=None) -> dict:
    folder = NOTES_DIR / paper["folder"]
    folder.mkdir(parents=True, exist_ok=True)

    pdf_path = folder / paper["pdf_name"]
    note_path = folder / paper["note_name"]

    result = {
        "folder": paper["folder"],
        "pdf_ok": False,
        "note_ok": False,
        "pdf_path": str(pdf_path),
        "note_path": str(note_path),
    }

    # skip re-download if valid PDF already exists
    if pdf_path.exists() and pdf_path.stat().st_size > 10_000:
        with pdf_path.open("rb") as f:
            if f.read(5) == b"%PDF-":
                result["pdf_ok"] = True
                if console is not None and USE_RICH:
                    console.print(f"  [cyan]PDF 已存在，跳过下载[/]: {pdf_path.name}")
                else:
                    print(f"  PDF 已存在，跳过下载: {pdf_path.name}")

    if not result["pdf_ok"]:
        if console is not None and USE_RICH:
            console.print(f"  下载: {paper['pdf_url']}")
        else:
            print(f"  下载: {paper['pdf_url']}")
        result["pdf_ok"] = download_pdf(paper["pdf_url"], pdf_path, console=console)

    note_path.write_text(build_note_markdown(paper), encoding="utf-8")
    result["note_ok"] = note_path.exists() and note_path.stat().st_size > 0

    if result["note_ok"]:
        try_export_pdf_from_markdown(note_path, console=console)

    return result


def main() -> int:
    console = Console() if USE_RICH else None
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    banner = "Medical CV · 论文归档初始化"
    if console:
        console.rule(f"[bold]{banner}[/]")
        console.print(f"目标目录: [bold]{NOTES_DIR}[/]\n")
    else:
        print("=" * 60)
        print(banner)
        print(f"目标目录: {NOTES_DIR}\n")

    results = []
    for i, paper in enumerate(PAPERS, 1):
        title = f"[{i}/{len(PAPERS)}] {paper['folder']}"
        if console:
            console.print(f"[bold magenta]{title}[/]")
        else:
            print(title)
        results.append(init_paper(paper, console=console))
        if console:
            console.print()
        else:
            print()

    # summary
    ok_pdf = sum(1 for r in results if r["pdf_ok"])
    ok_note = sum(1 for r in results if r["note_ok"])
    if console:
        console.rule("[bold]检查结果[/]")
        for r in results:
            pdf_flag = "[green]OK[/]" if r["pdf_ok"] else "[red]FAIL[/]"
            note_flag = "[green]OK[/]" if r["note_ok"] else "[red]FAIL[/]"
            console.print(f"• {r['folder']}\n    PDF={pdf_flag}  Note={note_flag}")
        console.print(f"\n合计: PDF {ok_pdf}/{len(results)} · Notes {ok_note}/{len(results)}")
    else:
        print("=" * 60)
        print("检查结果")
        for r in results:
            print(f"• {r['folder']}: PDF={'OK' if r['pdf_ok'] else 'FAIL'} Note={'OK' if r['note_ok'] else 'FAIL'}")
        print(f"合计: PDF {ok_pdf}/{len(results)} · Notes {ok_note}/{len(results)}")

    if ok_pdf != len(results) or ok_note != len(results):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
