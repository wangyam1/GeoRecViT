#!/usr/bin/env python3
"""PCA Feature Visualization & Subspace Manifold Alignment Analysis for GeoRecViT.

Produces two paper-ready diagnostics directly on frozen SAM 3 PE-ViT + DINOv3-SAT + GeoRecViT:
  1. DINOv3-style Top-3 RGB-PCA spatial feature maps comparing:
       - Input RGB & Ground Truth
       - Frozen SAM 3 PE-ViT `F_pe^(24)` (natural-image open-vocab expert)
       - Frozen DINOv3-SAT-493M `F_sat` (satellite structural expert)
       - Aligned manifold `inner_link(F_pe^(24) + sat_outer_link(F_sat))`
       - GeoRecViT Round-1 vs Round-3 late-ViT outputs + Feature Shift `|F^(3) - F^(1)|`
       - STE Uncertainty Map + Routed Top-M 24x24 windows + Round-1 vs Round-3 predictions
  2. Quantitative PCA Spectrum & Cross-Expert Subspace Alignment JSON:
       - Cumulative explained variance ratio (top-k PCs) of each representation
       - Mean cosine similarity & Principal Subspace Overlap between PE-ViT and DINOv3-SAT
         before vs after Stage 1 Inner-Loop manifold alignment.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from georecvit.data import build_rs_datasets, load_yaml_config
from georecvit.models import DINOv3Expert, GeoRecViT, GeoRecViTConfig, SAM3RecursiveBackbone
from georecvit.utils import load_georecvit_links


# Distinct color palette for up to 8 land-cover classes (OpenEarthMap / LoveDA)
CLASS_PALETTE = np.array(
    [
        [128, 0, 0],      # 0: bareland / background
        [0, 255, 36],     # 1: rangeland / building
        [148, 148, 148],  # 2: developed_space / road
        [255, 255, 255],  # 3: road / water
        [34, 97, 38],     # 4: tree / barren
        [0, 69, 255],     # 5: water / forest
        [75, 181, 73],    # 6: agriculture_land / agricultural
        [222, 31, 7],     # 7: building
    ],
    dtype=np.uint8,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PCA feature & subspace alignment analysis for GeoRecViT.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--sam3_ckpt", type=str, required=True)
    p.add_argument("--dinov3_dir", type=str, required=True)
    p.add_argument("--links_ckpt", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/content/georecvit_cache/datasets")
    p.add_argument("--num_vis", type=int, default=4, help="Number of qualitative PCA panel PNGs to save")
    p.add_argument("--num_stat_images", type=int, default=20, help="Images used for PCA spectrum stats")
    p.add_argument("--num_rounds", type=int, default=3)
    p.add_argument("--top_m_windows", type=int, default=3)
    p.add_argument("--precision", type=str, default="fp16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--output_dir", type=str, default="results/pca_analysis")
    return p.parse_args()


def feat_grid_to_pca_rgb(feat_hwc: torch.Tensor) -> np.ndarray:
    """Project `[H, W, C]` patch tokens onto top-3 PCA components -> uint8 RGB `[H, W, 3]`."""
    H, W, C = feat_hwc.shape
    x = feat_hwc.detach().float().reshape(H * W, C)
    x = x - x.mean(dim=0, keepdim=True)
    _, _, v = torch.pca_lowrank(x, q=3, center=False)
    proj = (x @ v[:, :3]).cpu().numpy().reshape(H, W, 3)
    # Robust 2nd-98th percentile contrast stretch per channel (DINOv2/v3 style)
    out = np.zeros_like(proj, dtype=np.float32)
    for c in range(3):
        lo, hi = np.percentile(proj[..., c], [2.0, 98.0])
        out[..., c] = np.clip((proj[..., c] - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (out * 255.0).astype(np.uint8)


def colorize_mask(mask_hw: np.ndarray, ignore_index: int = 255) -> np.ndarray:
    H, W = mask_hw.shape
    rgb = np.zeros((H, W, 3), dtype=np.uint8)
    valid = (mask_hw != ignore_index) & (mask_hw >= 0) & (mask_hw < len(CLASS_PALETTE))
    rgb[valid] = CLASS_PALETTE[mask_hw[valid]]
    return rgb


def heatmap_to_rgb(map_hw: np.ndarray) -> np.ndarray:
    """Simple magma-like false-color heatmap `[H, W]` in `[0, 1]` -> uint8 RGB."""
    v = np.clip(map_hw, 0.0, 1.0)
    r = np.clip(1.8 * v, 0.0, 1.0)
    g = np.clip(2.0 * (v - 0.35), 0.0, 1.0)
    b = np.clip(0.8 * np.sin(np.pi * v) + 0.3 * (1.0 - v), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


def pca_spectrum_stats(tokens_nc: torch.Tensor, max_k: int = 64) -> Dict[str, object]:
    """Compute PCA explained variance curve and effective rank (participation ratio)."""
    x = tokens_nc.float()
    x = x - x.mean(dim=0, keepdim=True)
    q = min(max_k, x.shape[0] - 1, x.shape[1])
    _, s, v = torch.pca_lowrank(x, q=q, center=False)
    eig = s.pow(2)
    total_var = x.pow(2).sum().clamp(min=1e-8)
    ratios = (eig / total_var).cpu().numpy()
    cum = np.cumsum(ratios)
    # Effective rank (participation ratio on top-q spectrum)
    p = ratios / max(float(ratios.sum()), 1e-8)
    eff_rank = float(np.exp(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum()))
    return {
        "top3_cum_var": round(float(cum[min(2, len(cum) - 1)]), 4),
        "top16_cum_var": round(float(cum[min(15, len(cum) - 1)]), 4),
        "top32_cum_var": round(float(cum[min(31, len(cum) - 1)]), 4),
        "top64_cum_var": round(float(cum[-1]), 4),
        "effective_rank_top64": round(eff_rank, 2),
        "basis_top16": v[:, : min(16, v.shape[1])],
    }


def subspace_overlap(v_a: torch.Tensor, v_b: torch.Tensor) -> float:
    """Mean squared principal cosine similarity (Grassmann projection overlap in [0, 1])."""
    k = min(v_a.shape[1], v_b.shape[1])
    m = v_a[:, :k].T @ v_b[:, :k]
    return round(float((m.pow(2).sum() / max(k, 1)).item()), 4)


def make_panel_grid(tiles: List[np.ndarray], tile_size: int = 256, cols: int = 4) -> Image.Image:
    rows = (len(tiles) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * tile_size, rows * tile_size), color=(20, 20, 20))
    for idx, arr in enumerate(tiles):
        r, c = divmod(idx, cols)
        im = Image.fromarray(arr).resize((tile_size, tile_size), Image.NEAREST if idx in (1, 6, 7) else Image.BILINEAR)
        canvas.paste(im, (c * tile_size, r * tile_size))
    return canvas


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    use_amp = device.type == "cuda" and args.precision != "fp32"

    from sam3.model_builder import build_sam3_image_model

    sam3_model = build_sam3_image_model(
        checkpoint_path=args.sam3_ckpt,
        load_from_HF=False,
        device=str(device),
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
    )
    if use_amp:
        for mod in sam3_model.backbone.modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Embedding)):
                mod.to(dtype=amp_dtype)

    sam3_wrapper = SAM3RecursiveBackbone(
        sam3_model=sam3_model,
        split_block_idx=int(m_cfg.get("split_block_idx", 24)),
        resolution=int(ds_cfg.get("sam3_resolution", 1008)),
        use_grad_checkpoint=False,
        class_chunk_size=2,
    )
    dinov3 = DINOv3Expert(
        model_path=args.dinov3_dir,
        input_size=int(ds_cfg.get("dinov3_resolution", 512)),
        dtype=amp_dtype if use_amp else torch.float32,
    )
    model_cfg = GeoRecViTConfig(
        num_classes=int(ds_cfg["num_classes"]),
        pe_dim=sam3_wrapper.embed_dim,
        sat_dim=dinov3.embed_dim,
        decoder_dim=sam3_wrapper.d_model,
        window_size=sam3_wrapper.window_size,
        num_recursive_rounds=args.num_rounds,
        sparse_top_windows=args.top_m_windows,
        use_sam3_detr=True,
        detr_round1_only=True,
        use_dinov3=True,
        use_inner_link=True,
        use_feedback_loop=True,
    )
    model = GeoRecViT(sam3_wrapper, dinov3, model_cfg).to(device)
    load_georecvit_links(model, args.links_ckpt, strict=False)
    model.eval()

    _, val_ds = build_rs_datasets(
        cfg,
        cache_dir=args.cache_dir,
        max_train_samples=4,
        max_val_samples=max(args.num_stat_images, args.num_vis),
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    class_prompts = ds_cfg["class_prompts"]
    ignore_index = int(ds_cfg.get("ignore_index", 255))

    # Subsample patch tokens per image for global PCA subspace statistics
    buf_pe, buf_sat, buf_sat_proj, buf_aligned = [], [], [], []
    cos_raw_list, cos_proj_list, cos_aligned_list = [], [], []
    shift_active_list, shift_inactive_list = [], []

    with torch.inference_mode():
        for i in range(len(val_ds)):
            sample = val_ds[i]
            img = sample["image"].unsqueeze(0).to(device)
            mask = sample["mask"].numpy()

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                f_pe_mid = model.sam3_wrapper.forward_early_vit(img)      # [1, 72, 72, 1024]
                f_sat = model.dinov3_expert(img).to(dtype=f_pe_mid.dtype) # [1, 72, 72, 1024]
                sat_ungated = model.sat_outer_link(f_sat, apply_gate=False)
                sat_gated = model.sat_outer_link(f_sat, apply_gate=True)
                aligned_mid = model.inner_link(f_pe_mid + sat_gated)

                out = model(
                    img,
                    class_prompts=class_prompts,
                    num_rounds=args.num_rounds,
                    top_m_windows=args.top_m_windows,
                    return_routing=True,
                )

            # Cosine alignment before vs after Stage 1 Inner-Loop
            cos_raw_list.append(float(F.cosine_similarity(f_pe_mid.float(), f_sat.float(), dim=-1).mean().item()))
            cos_proj_list.append(float(F.cosine_similarity(f_pe_mid.float(), sat_ungated.float(), dim=-1).mean().item()))
            cos_aligned_list.append(float(F.cosine_similarity(f_pe_mid.float(), aligned_mid.float(), dim=-1).mean().item()))

            # Sample every 4th patch token (18x18=324 tokens/img) for global subspace PCA
            buf_pe.append(f_pe_mid[0, ::4, ::4].float().reshape(-1, f_pe_mid.shape[-1]).cpu())
            buf_sat.append(f_sat[0, ::4, ::4].float().reshape(-1, f_sat.shape[-1]).cpu())
            buf_sat_proj.append(sat_ungated[0, ::4, ::4].float().reshape(-1, sat_ungated.shape[-1]).cpu())
            buf_aligned.append(aligned_mid[0, ::4, ::4].float().reshape(-1, aligned_mid.shape[-1]).cpu())

            # Measure Round-1 -> Round-R logit shift inside vs outside routed windows
            z1 = out["round_logits"][0].float()
            zr = out["round_logits"][-1].float()
            delta_l2 = (zr - z1).norm(dim=1, keepdim=True)  # [1, 1, 288, 288]
            if out.get("active_masks_hi"):
                u_mask = torch.stack(out["active_masks_hi"], dim=0).amax(dim=0) > 0.5
                if u_mask.any():
                    shift_active_list.append(float(delta_l2[u_mask].mean().item()))
                if (~u_mask).any():
                    shift_inactive_list.append(float(delta_l2[~u_mask].mean().item()))

            # Save qualitative 8-panel PCA + Routing + Prediction figure
            if i < args.num_vis:
                rgb_img = (sample["image"].permute(1, 2, 0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
                gt_rgb = colorize_mask(mask, ignore_index=ignore_index)
                pca_pe = feat_grid_to_pca_rgb(f_pe_mid[0])
                pca_sat = feat_grid_to_pca_rgb(f_sat[0])
                pca_aligned = feat_grid_to_pca_rgb(aligned_mid[0])

                # Uncertainty heatmap overlaid with selected window boundary brightness
                routing0 = out["routing_history"][0]
                u_map = routing0.uncertainty_map[0, 0].float().cpu().numpy()
                u_norm = (u_map - u_map.min()) / max(u_map.max() - u_map.min(), 1e-6)
                act_m = routing0.active_mask[0, :, :, 0].float().cpu().numpy()
                u_vis = heatmap_to_rgb(0.65 * u_norm + 0.35 * act_m)

                pred_r1 = F.interpolate(z1, size=mask.shape, mode="bilinear", align_corners=False).argmax(dim=1)[0].cpu().numpy()
                pred_rr = F.interpolate(zr, size=mask.shape, mode="bilinear", align_corners=False).argmax(dim=1)[0].cpu().numpy()
                pred_r1_rgb = colorize_mask(pred_r1, ignore_index=ignore_index)
                pred_rr_rgb = colorize_mask(pred_rr, ignore_index=ignore_index)

                # 2x4 panel: [RGB, GT, PCA(SAM3), PCA(DINOv3-SAT)] / [PCA(Aligned), Uncertainty+TopM, Pred R1, Pred R3]
                panel = make_panel_grid(
                    [rgb_img, gt_rgb, pca_pe, pca_sat, pca_aligned, u_vis, pred_r1_rgb, pred_rr_rgb],
                    tile_size=256,
                    cols=4,
                )
                panel.save(out_dir / f"pca_panel_val_{i:02d}.png")

    t_pe = torch.cat(buf_pe, dim=0)
    t_sat = torch.cat(buf_sat, dim=0)
    t_sat_proj = torch.cat(buf_sat_proj, dim=0)
    t_aligned = torch.cat(buf_aligned, dim=0)

    s_pe = pca_spectrum_stats(t_pe)
    s_sat = pca_spectrum_stats(t_sat)
    s_sat_proj = pca_spectrum_stats(t_sat_proj)
    s_aligned = pca_spectrum_stats(t_aligned)

    report = {
        "dataset": ds_cfg["name"],
        "num_stat_images": len(val_ds),
        "num_tokens_analyzed": int(t_pe.shape[0]),
        "panel_layout_2x4": [
            "row0_col0: Input Satellite RGB",
            "row0_col1: Ground Truth Mask",
            "row0_col2: PCA-RGB Frozen SAM 3 PE-ViT F_pe^(24)",
            "row0_col3: PCA-RGB Frozen DINOv3-SAT-493M F_sat",
            "row1_col0: PCA-RGB Stage-1 Aligned Manifold (PE-ViT + OuterLink + InnerLink)",
            "row1_col1: STE Hybrid Uncertainty + Top-M Routed Windows",
            "row1_col2: GeoRecViT Round-1 Prediction",
            "row1_col3: GeoRecViT Round-3 Prediction",
        ],
        "pca_spectrum": {
            "sam3_pe_vit_mid": {k: v for k, v in s_pe.items() if k != "basis_top16"},
            "dinov3_sat_raw": {k: v for k, v in s_sat.items() if k != "basis_top16"},
            "dinov3_sat_outer_projected": {k: v for k, v in s_sat_proj.items() if k != "basis_top16"},
            "georecvit_aligned_manifold": {k: v for k, v in s_aligned.items() if k != "basis_top16"},
        },
        "cross_expert_subspace_alignment": {
            "mean_cosine_raw_pe_vs_sat": round(float(np.mean(cos_raw_list)), 4),
            "mean_cosine_pe_vs_outer_projected_sat": round(float(np.mean(cos_proj_list)), 4),
            "mean_cosine_pe_vs_inner_aligned": round(float(np.mean(cos_aligned_list)), 4),
            "top16_grassmann_overlap_raw_pe_vs_sat": subspace_overlap(s_pe["basis_top16"], s_sat["basis_top16"]),
            "top16_grassmann_overlap_pe_vs_outer_projected_sat": subspace_overlap(s_pe["basis_top16"], s_sat_proj["basis_top16"]),
            "top16_grassmann_overlap_pe_vs_inner_aligned": subspace_overlap(s_pe["basis_top16"], s_aligned["basis_top16"]),
        },
        "recursive_logit_shift_l2": {
            "mean_shift_inside_active_windows": round(float(np.mean(shift_active_list)), 5) if shift_active_list else 0.0,
            "mean_shift_outside_active_windows": round(float(np.mean(shift_inactive_list)), 5) if shift_inactive_list else 0.0,
        },
    }
    report_path = out_dir / "pca_subspace_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2), flush=True)
    print(f"[pca] Saved PCA report & {args.num_vis} panel PNGs to {out_dir}", flush=True)


if __name__ == "__main__":
    main()
