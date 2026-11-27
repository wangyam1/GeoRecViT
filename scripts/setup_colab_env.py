#!/usr/bin/env python3
"""Colab VM setup script: installs deps, pulls ModelScope weights & verifies HF RS datasets.

Runs strictly on the remote Colab VM (e.g. via `colab exec -s georecvit -f scripts/setup_colab_env.py`).
All large weights and datasets stay under `/content/georecvit_cache/` (Colab fast SSD).
"""

from __future__ import annotations

import os
from pathlib import Path
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


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Check GPU & PyTorch
    import torch
    print(f"[setup] PyTorch={torch.__version__}, CUDA={torch.cuda.is_available()}", flush=True)
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(
            f"[setup] GPU={props.name}, VRAM={props.total_memory / 1e9:.1f} GB, "
            f"bf16={torch.cuda.is_bf16_supported()}",
            flush=True,
        )

    # 2. Clone/update SAM 3 and GeoRecViT repos
    if not SAM3_REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/facebookresearch/sam3.git", str(SAM3_REPO_DIR)])
    if not GEORECVIT_REPO_DIR.exists():
        run(["git", "clone", "--depth", "1", "https://github.com/wangyam1/GeoRecViT.git", str(GEORECVIT_REPO_DIR)])
    else:
        run(["git", "pull", "--ff-only"], cwd=GEORECVIT_REPO_DIR)

    # 3. Install lightweight Python dependencies
    run([
        sys.executable, "-m", "pip", "install", "-q",
        "modelscope>=1.20.0", "datasets>=2.20.0", "iopath", "timm", "einops", "pyyaml", "transformers>=4.56.0",
    ])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(SAM3_REPO_DIR)])
    run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "-e", str(GEORECVIT_REPO_DIR)])

    # 4. Download ModelScope weights (no HF gated wait!)
    from modelscope import snapshot_download

    sam3_dir = WEIGHTS_DIR / "sam3"
    if not (sam3_dir / "sam3.pt").exists():
        print("[setup] Downloading facebook/sam3 from ModelScope...", flush=True)
        snapshot_download(
            "facebook/sam3",
            local_dir=str(sam3_dir),
            allow_patterns=["sam3.pt", "config.json", "*.txt", "*.json"],
        )
    else:
        print(f"[setup] Cached SAM 3 found: {sam3_dir / 'sam3.pt'}", flush=True)

    dinov3_sat_dir = WEIGHTS_DIR / "dinov3-vitl16-pretrain-sat493m"
    if not (dinov3_sat_dir / "model.safetensors").exists():
        print("[setup] Downloading facebook/dinov3-vitl16-pretrain-sat493m from ModelScope...", flush=True)
        snapshot_download(
            "facebook/dinov3-vitl16-pretrain-sat493m",
            local_dir=str(dinov3_sat_dir),
            allow_patterns=["*.safetensors", "*.json"],
        )
    else:
        print(f"[setup] Cached DINOv3-SAT found: {dinov3_sat_dir}", flush=True)

    # 5. Pull and verify LoveDA & OpenEarthMap datasets + print exact pixel label ranges
    import numpy as np
    from datasets import load_dataset

    for name, repo, split, mask_col in [
        ("loveda", "chloechia/loveda", "validation", "label"),
        ("openearthmap", "EVER-Z/open_earth_map", "val", "mask"),
    ]:
        print(f"[setup] Pulling {name} ({repo}, split={split}) into {DATA_DIR}...", flush=True)
        ds = load_dataset(repo, split=split, cache_dir=str(DATA_DIR))
        uniq = set()
        for i in range(min(16, len(ds))):
            arr = np.asarray(ds[i][mask_col])
            if arr.ndim == 3:
                arr = arr[..., 0]
            uniq.update(int(x) for x in np.unique(arr))
        print(
            f"[setup] {name}: val_len={len(ds)}, raw_mask_unique_sample={sorted(uniq)}",
            flush=True,
        )

    print("[setup] Environment, ModelScope weights, and RS datasets ready!", flush=True)


if __name__ == "__main__":
    main()
