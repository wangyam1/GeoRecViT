#!/usr/bin/env python3
"""Colab VM setup script: fast aria2c ModelScope weights + Zenodo LoveDA + HF OpenEarthMap.

All large weights and datasets stay strictly on Colab SSD (`/content/georecvit_cache/`).
"""

from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys


CACHE_ROOT = Path(os.environ.get("GEORECVIT_CACHE", "/content/georecvit_cache"))
WEIGHTS_DIR = CACHE_ROOT / "weights"
DATA_DIR = CACHE_ROOT / "datasets"
SAM3_REPO_DIR = Path("/content/sam3")
GEORECVIT_REPO_DIR = Path("/content/GeoRecViT")


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"[setup] $ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def fast_download(url: str, dest_dir: Path, filename: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename
    if target.exists() and target.stat().st_size > 0:
        print(f"[setup] Cached: {target}", flush=True)
        return
    if shutil.which("aria2c"):
        run([
            "aria2c", "-x", "16", "-s", "16", "-k", "2M",
            "--file-allocation=none", "--summary-interval=20", "--console-log-level=notice",
            "-d", str(dest_dir), "-o", filename, url,
        ])
    else:
        run(["curl", "-L", "-o", str(target), url])


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    import torch
    print(f"[setup] PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"[setup] GPU={props.name}, VRAM={props.total_memory / 1e9:.1f} GB, "
            f"bf16={torch.cuda.is_bf16_supported()}",
            flush=True,
        )

    if not shutil.which("aria2c"):
        run(["apt-get", "update", "-qq"])
        run(["apt-get", "install", "-y", "-qq", "aria2", "unzip"])

    if not SAM3_REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/sam3.git", str(SAM3_REPO_DIR)])
    if not GEORECVIT_REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/wangyam1/GeoRecViT.git", str(GEORECVIT_REPO_DIR)])
    else:
        run(["git", "pull", "--ff-only"], cwd=GEORECVIT_REPO_DIR)

    run([
        sys.executable, "-m", "pip", "install", "-q",
        "modelscope>=1.20.0", "datasets>=2.20.0", "iopath", "timm", "einops", "pyyaml",
        "transformers>=4.56.0", "setuptools<81", "ftfy", "regex",
    ])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "-e", str(SAM3_REPO_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--no-build-isolation", "-e", str(GEORECVIT_REPO_DIR)])

    # 1. ModelScope weights via fast 16-connection aria2c
    sam3_dir = WEIGHTS_DIR / "sam3"
    fast_download("https://www.modelscope.cn/models/facebook/sam3/resolve/master/sam3.pt", sam3_dir, "sam3.pt")
    for f in ["config.json", "processor_config.json", "tokenizer.json", "vocab.json", "merges.txt"]:
        fast_download(f"https://www.modelscope.cn/models/facebook/sam3/resolve/master/{f}", sam3_dir, f)

    dinov3_dir = WEIGHTS_DIR / "dinov3-vitl16-pretrain-sat493m"
    for f in ["config.json", "configuration.json", "preprocessor_config.json", "model.safetensors"]:
        fast_download(
            f"https://www.modelscope.cn/models/facebook/dinov3-vitl16-pretrain-sat493m/resolve/master/{f}",
            dinov3_dir, f,
        )

    # 2. LoveDA Official Zenodo Val.zip (+ optional Train.zip when DOWNLOAD_LOVEDA_TRAIN=1)
    loveda_dir = DATA_DIR / "loveda"
    if not (loveda_dir / "Val").exists():
        fast_download("https://zenodo.org/records/5706578/files/Val.zip?download=1", loveda_dir, "Val.zip")
        run(["unzip", "-q", "-o", str(loveda_dir / "Val.zip"), "-d", str(loveda_dir)])
        (loveda_dir / "Val.zip").unlink(missing_ok=True)
    if os.environ.get("DOWNLOAD_LOVEDA_TRAIN", "0") == "1" and not (loveda_dir / "Train").exists():
        fast_download("https://zenodo.org/records/5706578/files/Train.zip?download=1", loveda_dir, "Train.zip")
        run(["unzip", "-q", "-o", str(loveda_dir / "Train.zip"), "-d", str(loveda_dir)])
        (loveda_dir / "Train.zip").unlink(missing_ok=True)

    # 3. OpenEarthMap (Parquet, fast!)
    from datasets import load_dataset
    ds = load_dataset("EVER-Z/open_earth_map", split="val", cache_dir=str(DATA_DIR))
    print(f"[setup] OpenEarthMap val ready: {len(ds)} images", flush=True)
    print("[setup] All weights & datasets ready!", flush=True)


if __name__ == "__main__":
    main()
