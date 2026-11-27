# GeoRecViT: Heterogeneous Recursive ViT with SAM 3 & DINOv3-SAT for Remote Sensing Segmentation

**GeoRecViT** brings the latent-space closed-loop recursion paradigm of **RecursiveMAS (NeurIPS 2026)** into high-resolution **2D Vision Transformers** for challenging, non-saturated remote sensing segmentation benchmarks (**LoveDA** and **OpenEarthMap**).

## Motivation & Peer Positioning

Recent 2026 SAM 3 remote sensing adaptations (`SegEarth-OV3`, `ProC-SAM3`, `ROSE-SAM3`) operate strictly as **single-pass prompt tuning** outside SAM 3's frozen natural-image `PE-ViT`, leaving overhead domain shift (tiny vehicles, dense shadowed rooftops, urban-rural texture gaps) unresolved. Meanwhile, Meta's satellite-pretrained **`DINOv3-ViT-L/16-SAT493M`** excels at remote sensing geometry but lacks open-vocabulary concept segmentation heads.

**GeoRecViT** bridges both worlds via 4 design principles:
1. **Heterogeneous Latent Visual Collaboration**: Pairs frozen **SAM 3 (`PE-ViT` + `DETR` + `UniversalSegmentationHead`)** with frozen **`DINOv3-SAT-493M`** as complementary visual experts in continuous latent space.
2. **Closed-Loop ViT Recursion**: Feeds Round-$r$ segmentation hypotheses, prediction entropy, and cross-expert disagreement back into SAM 3's late ViT stage (`blocks[24:32]`) for iterative top-down refinement ($R = 1 \to 2 \to 3$).
3. **Uncertainty-Gated Sparse Native Window Routing**: Exploits SAM 3's exact `72×72` grid divisibility by `window_size=24` (`3×3 = 9` native windows) to recompute ONLY the **Top-$M$ most uncertain windows** (e.g. $M=3/9 = 33.3\%$) across frozen `blocks[24:30]` with **zero 2D RoPE distortion**, followed by the single global `block[31]`.
4. **Inner-Outer Loop Two-Stage Training (100% Frozen Backbones)**: Only ~12M lightweight `SpatialInnerLink` / `SpatialOuterLink` / `FeedbackEncoderLink` parameters are trained—preserving SAM 3's open-vocabulary capabilities while fitting comfortably on a single Colab GPU.

## Target Benchmarks (Non-Saturated, SOTA < 70% mIoU)

| Dataset | HF Repo | Classes | Supervised SOTA (mIoU) | SAM 3 OVSS (`SegEarth-OV3`) | Primary Challenges |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **LoveDA** | `chloechia/loveda` | 7 | 55.3%–56.2% *(D2LS ICCV'25 / IQ2Former)* | 47.4% | Urban↔Rural domain shift, dense rooftops, barren/agri confusion |
| **OpenEarthMap** | `EVER-Z/open_earth_map` | 8 | 61.0%–68.0% *(UNetFormer / Mask2Former)* | 42.9% | 97 global regions across 6 continents, cross-geography shift |

## Quickstart on Colab (via `colab` CLI)

All large model weights (`facebook/sam3`, `facebook/dinov3-vitl16-pretrain-sat493m` via **ModelScope**) and datasets (`LoveDA`, `OpenEarthMap` via **HuggingFace**) are downloaded directly on the Colab VM SSD (`/content/georecvit_cache/`). This repo stays lightweight with code, configs, and JSON metrics only.

```bash
# 1. Provision GPU session
colab new -s georecvit --gpu A100

# 2. Setup environment, pull ModelScope weights & verify RS datasets
colab exec -s georecvit -f scripts/setup_colab_env.py

# 3. Baseline 0: Zero-Shot Frozen SAM 3 (semantic + SegEarth-OV3 dual-head)
colab ssh -s georecvit -- "cd /content/GeoRecViT && python scripts/eval_sam3_baseline.py \
  --config configs/loveda.yaml \
  --sam3_ckpt /content/georecvit_cache/weights/sam3/sam3.pt \
  --output_json /content/georecvit_cache/results/loveda_baseline0.json"

# 4. Stage 1: Inner-Loop Manifold Warm-Start
colab ssh -s georecvit -- "cd /content/GeoRecViT && python scripts/train_inner.py \
  --config configs/loveda.yaml \
  --sam3_ckpt /content/georecvit_cache/weights/sam3/sam3.pt \
  --dinov3_dir /content/georecvit_cache/weights/dinov3-vitl16-pretrain-sat493m"

# 5. Stage 2: Outer-Loop Closed-Loop Multi-Round Training (R=3, M=3/9 sparse windows)
colab ssh -s georecvit -- "cd /content/GeoRecViT && python scripts/train_outer.py \
  --config configs/loveda.yaml \
  --sam3_ckpt /content/georecvit_cache/weights/sam3/sam3.pt \
  --dinov3_dir /content/georecvit_cache/weights/dinov3-vitl16-pretrain-sat493m \
  --inner_ckpt /content/georecvit_cache/checkpoints/inner/georecvit_links_inner_warmstart.pt"

# 6. Always stop VM when idle to conserve compute units!
colab stop -s georecvit
```

## Local Unit Tests (CPU, <5s, No Large Downloads)

```bash
pip install -e . --no-deps
pytest -q
```
