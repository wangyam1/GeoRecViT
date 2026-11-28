#!/usr/bin/env python3
"""Zero-Retraining Supplementary / Appendix Evaluation Suite (Appendices A–E).

Runs on any trained GeoRecViT checkpoint and produces publication-ready JSON + PNGs:
  - Appendix A: Test-Time Depth Extrapolation (`R = 1, 2, 3, 4, 5`) stability curve
  - Appendix B: Routing Strategy Ablation (`hybrid` vs `entropy_only` vs `disagreement_only` vs `random`)
  - Appendix C: LoveDA Urban vs Rural separate domain breakdown (when dataset == loveda)
  - Appendix D: Exact per-module parameter breakdown & peak GPU VRAM measurement
  - Appendix E: Pixel-level Green/Red Error-Correction Panels (`Round 1 -> Round 3`) & Failure Cases
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from georecvit.data import build_rs_datasets, load_yaml_config
from georecvit.models import (
    DINOv3Expert,
    GeoRecViT,
    GeoRecViTConfig,
    SAM2RecursiveBackbone,
    SAM3RecursiveBackbone,
)
from georecvit.utils import SegmentationMeter, extract_boundary_mask, load_georecvit_links


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Appendix A-E evaluation reports & correction figures.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--expert_a", type=str, default="sam3", choices=["sam3", "sam2"])
    p.add_argument("--sam3_ckpt", type=str, default="")
    p.add_argument("--sam2_dir", type=str, default="")
    p.add_argument("--dinov3_dir", type=str, required=True)
    p.add_argument("--links_ckpt", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/content/georecvit_cache/datasets")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--bidir_sat_recursion", action="store_true")
    p.add_argument("--max_val_samples", type=int, default=120)
    p.add_argument("--num_vis", type=int, default=4)
    p.add_argument("--top_m_windows", type=int, default=3)
    p.add_argument("--class_chunk_size", type=int, default=4)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def _save_correction_panel(
    img_01: torch.Tensor,
    gt: torch.Tensor,
    pred_r1: torch.Tensor,
    pred_r3: torch.Tensor,
    active_union: torch.Tensor,
    ignore_index: int,
    out_path: Path,
    panel_size: int = 256,
) -> dict:
    """Save 1x4 panel: RGB | Active Windows | Error-Correction Map (Green=Fixed, Red=Broken) | Disagreement Diff."""
    rgb = (img_01.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    gt_np = gt.cpu().numpy()
    p1_np = pred_r1.cpu().numpy()
    p3_np = pred_r3.cpu().numpy()
    act_np = active_union.cpu().numpy() > 0.5

    valid = gt_np != ignore_index
    r1_ok = (p1_np == gt_np) & valid
    r3_ok = (p3_np == gt_np) & valid
    fixed = (~r1_ok) & r3_ok & valid
    broken = r1_ok & (~r3_ok) & valid
    still_err = (~r1_ok) & (~r3_ok) & valid
    changed = (p1_np != p3_np)

    # Panel 2: RGB + yellow overlay on STE active windows
    act_vis = rgb.copy().astype(np.float32)
    act_vis[act_np] = 0.55 * act_vis[act_np] + 0.45 * np.array([255.0, 230.0, 40.0])
    act_vis = np.clip(act_vis, 0, 255).astype(np.uint8)

    # Panel 3: Correction heatmap (Green=fixed by R3, Red=regressed, DarkGray=unchanged error, Light=correct)
    corr = np.full_like(rgb, 235)
    corr[still_err] = [95, 95, 110]
    corr[fixed] = [20, 210, 75]
    corr[broken] = [235, 45, 45]
    corr[~valid] = [30, 30, 30]

    # Panel 4: Changed pixels overlay on RGB (proving zero changes outside active windows!)
    chg_vis = (rgb.astype(np.float32) * 0.45).astype(np.uint8)
    chg_vis[changed & fixed] = [30, 245, 90]
    chg_vis[changed & broken] = [255, 50, 50]
    chg_vis[changed & (~fixed) & (~broken)] = [80, 180, 255]

    tiles = [
        Image.fromarray(arr).resize((panel_size, panel_size), Image.BILINEAR)
        for arr in (rgb, act_vis, corr, chg_vis)
    ]
    canvas = Image.new("RGB", (panel_size * 4, panel_size))
    for i, t in enumerate(tiles):
        canvas.paste(t, (i * panel_size, 0))
    canvas.save(out_path)
    return {
        "fixed_pixels": int(fixed.sum()),
        "broken_pixels": int(broken.sum()),
        "net_fixed_pixels": int(fixed.sum()) - int(broken.sum()),
        "changed_outside_active": int((changed & (~act_np)).sum()),
    }


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    use_amp = device.type == "cuda" and args.precision != "fp32"

    if args.expert_a == "sam2":
        sam3_wrapper = SAM2RecursiveBackbone.from_pretrained(
            model_path=args.sam2_dir,
            resolution=1024,
            use_grad_checkpoint=False,
            dtype=amp_dtype if use_amp else torch.float32,
        )
    else:
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
            class_chunk_size=args.class_chunk_size,
        )
    dinov3 = DINOv3Expert(
        model_path=args.dinov3_dir,
        input_size=int(ds_cfg.get("dinov3_resolution", 512)),
        target_grid_hw=sam3_wrapper.grid_hw,
        split_layer_idx=18,
        use_grad_checkpoint=False,
        dtype=amp_dtype if use_amp else torch.float32,
    )
    model_cfg = GeoRecViTConfig(
        num_classes=int(ds_cfg["num_classes"]),
        pe_dim=sam3_wrapper.embed_dim,
        sat_dim=dinov3.embed_dim,
        decoder_dim=sam3_wrapper.d_model,
        window_size=sam3_wrapper.window_size,
        num_recursive_rounds=5,
        sparse_top_windows=args.top_m_windows,
        use_sam3_detr=bool(getattr(sam3_wrapper, "has_detr", True)) and bool(m_cfg.get("use_sam3_detr", True)),
        detr_round1_only=True,
        use_dinov3=True,
        use_inner_link=True,
        use_feedback_loop=True,
        bidir_sat_recursion=bool(args.bidir_sat_recursion),
    )
    model = GeoRecViT(sam3_wrapper, dinov3, model_cfg).to(device)
    load_georecvit_links(model, args.links_ckpt, strict=False)
    model.eval()

    # Appendix D: Exact per-module parameter breakdown
    module_params = {
        "frozen_sam3": sum(p.numel() for p in sam3_wrapper.parameters()),
        "frozen_dinov3_sat": sum(p.numel() for p in dinov3.parameters()),
    }
    for name, mod in model.trainable_modules().items():
        module_params[f"trainable_{name}"] = sum(p.numel() for p in mod.parameters())
    module_params["total_trainable"] = sum(p.numel() for p in model.trainable_parameters())
    module_params["total_params"] = sum(p.numel() for p in model.parameters())

    _, val_ds = build_rs_datasets(
        cfg, cache_dir=args.cache_dir, max_train_samples=4,
        max_val_samples=(args.max_val_samples if args.max_val_samples > 0 else None),
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    class_prompts = ds_cfg["class_prompts"]
    class_names = ds_cfg["class_names"]
    ignore_index = int(ds_cfg.get("ignore_index", 255))

    # Appendix A: Test-Time Depth Extrapolation R = 1, 2, 3, 4, 5 (+ Appendix E panels & Appendix F embedding trajectories!)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    extrap_rounds = [1, 2, 3, 4, 5]
    extrap_meters = {r: SegmentationMeter(len(class_names), class_names, ignore_index) for r in extrap_rounds}
    extrap_bnd = {r: SegmentationMeter(len(class_names), class_names, ignore_index) for r in extrap_rounds}
    extrap_act = {r: SegmentationMeter(len(class_names), class_names, ignore_index) for r in extrap_rounds}
    # Appendix F (mirrors RecursiveMAS Fig. 7): active-window class-distribution embeddings across R=1,2,3 vs GT
    num_cls = len(class_names)
    emb_gt_vecs: list[np.ndarray] = []
    emb_round_vecs: dict[int, list[np.ndarray]] = {1: [], 2: [], 3: []}
    vis_stats = []
    vis_indices = set(np.linspace(0, max(len(val_ds) - 1, 0), min(args.num_vis, len(val_ds)), dtype=int).tolist())

    t0_extrap = time.perf_counter()
    with torch.inference_mode():
        for idx, batch in enumerate(tqdm(loader, desc="AppendixA(R=1..5)")):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    images, class_prompts=class_prompts, num_rounds=5,
                    top_m_windows=args.top_m_windows, routing_mode="hybrid",
                )
            bmask = extract_boundary_mask(masks, ignore_index=ignore_index, radius=4)
            b_targets = torch.where(bmask, masks, torch.full_like(masks, ignore_index))
            u = torch.stack(out["active_masks_hi"][:2], dim=0).amax(dim=0)
            u_up = F.interpolate(u, size=masks.shape[-2:], mode="nearest").squeeze(1) > 0.5
            act_targets = torch.where(u_up, masks, torch.full_like(masks, ignore_index))

            preds = {}
            probs_r: dict[int, torch.Tensor] = {}
            for r in extrap_rounds:
                lg_up = F.interpolate(out["round_logits"][r - 1].float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
                pred = lg_up.argmax(dim=1)
                preds[r] = pred
                if r in (1, 2, 3):
                    probs_r[r] = F.softmax(lg_up, dim=1)
                extrap_meters[r].update(pred, masks)
                extrap_bnd[r].update(pred, b_targets)
                extrap_act[r].update(pred, act_targets)

            # Collect 3x3 native-window pooled class distributions for active windows (mirrors RecursiveMAS Fig. 7)
            valid_m = masks[0] != ignore_index
            gt_safe = torch.where(valid_m, masks[0], torch.zeros_like(masks[0]))
            gt_oh = F.one_hot(gt_safe, num_classes=num_cls).permute(2, 0, 1).float() * valid_m.unsqueeze(0).float()
            H_img, W_img = masks.shape[-2:]
            gh, gw = H_img // 3, W_img // 3
            for wy in range(3):
                for wx in range(3):
                    ys, ye = wy * gh, (wy + 1) * gh if wy < 2 else H_img
                    xs, xe = wx * gw, (wx + 1) * gw if wx < 2 else W_img
                    if u_up[0, ys:ye, xs:xe].float().mean().item() < 0.5:
                        continue
                    v_patch = valid_m[ys:ye, xs:xe]
                    denom = v_patch.float().sum().item()
                    if denom < 32.0:
                        continue
                    gt_v = (gt_oh[:, ys:ye, xs:xe].sum(dim=(1, 2)) / denom).cpu().numpy()
                    emb_gt_vecs.append(gt_v)
                    for r in (1, 2, 3):
                        pr_v = ((probs_r[r][0, :, ys:ye, xs:xe] * v_patch.unsqueeze(0).float()).sum(dim=(1, 2)) / denom).cpu().numpy()
                        emb_round_vecs[r].append(pr_v)

            if idx in vis_indices:
                stat = _save_correction_panel(
                    images[0], masks[0], preds[1][0], preds[3][0], u_up[0], ignore_index,
                    out_dir / f"correction_panel_{idx:03d}.png",
                )
                stat["val_idx"] = idx
                vis_stats.append(stat)
    peak_vram_gb = round(torch.cuda.max_memory_allocated(device) / (1024 ** 3), 2) if device.type == "cuda" else 0.0
    extrap_latency_ms = round(1000.0 * (time.perf_counter() - t0_extrap) / max(len(val_ds), 1), 2)

    # Fit shared 2D PCA on GT + R1..R3 active-window class distributions (RecursiveMAS Fig. 7 mirror)
    appendix_f = {}
    if len(emb_gt_vecs) >= 10:
        gt_mat = np.stack(emb_gt_vecs, axis=0)
        r_mats = {r: np.stack(emb_round_vecs[r], axis=0) for r in (1, 2, 3)}
        stacked = np.concatenate([gt_mat, r_mats[1], r_mats[2], r_mats[3]], axis=0)
        mean_vec = stacked.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(stacked - mean_vec, full_matrices=False)
        basis2 = vt[:2].T  # [C, 2]
        gt_2d = (gt_mat - mean_vec) @ basis2
        appendix_f = {
            "num_active_windows": int(gt_mat.shape[0]),
            "gt_2d": np.round(gt_2d, 5).tolist(),
            "rounds": {},
        }
        for r in (1, 2, 3):
            r_2d = (r_mats[r] - mean_vec) @ basis2
            l2_dist = float(np.linalg.norm(r_mats[r] - gt_mat, axis=1).mean())
            kl_div = float(np.mean(np.sum(gt_mat * np.log((gt_mat + 1e-6) / (r_mats[r] + 1e-6)), axis=1)))
            appendix_f["rounds"][f"R={r}"] = {
                "pred_2d": np.round(r_2d, 5).tolist(),
                "mean_l2_to_gt": round(l2_dist, 5),
                "mean_kl_gt_to_pred": round(kl_div, 5),
                "centroid_shift_2d": round(float(np.linalg.norm(r_2d.mean(axis=0) - gt_2d.mean(axis=0))), 5),
            }

    appendix_a = {
        "top_m_windows": args.top_m_windows,
        "avg_latency_ms_at_R5": extrap_latency_ms,
        "peak_eval_vram_gb": peak_vram_gb,
        "by_round": {f"R={r}": extrap_meters[r].compute() for r in extrap_rounds},
        "boundary_by_round_mIoU": {f"R={r}": extrap_bnd[r].compute()["mIoU"] for r in extrap_rounds},
        "active_window_by_round_mIoU": {f"R={r}": extrap_act[r].compute()["mIoU"] for r in extrap_rounds},
    }

    # Appendix B: Routing Strategy Ablation (Random vs Entropy-Only vs Disagreement-Only vs Hybrid STE)
    appendix_b = {}
    for mode in ("random", "entropy_only", "disagreement_only", "hybrid"):
        torch.manual_seed(42)
        m1 = SegmentationMeter(len(class_names), class_names, ignore_index)
        m3 = SegmentationMeter(len(class_names), class_names, ignore_index)
        b1 = SegmentationMeter(len(class_names), class_names, ignore_index)
        b3 = SegmentationMeter(len(class_names), class_names, ignore_index)
        a1 = SegmentationMeter(len(class_names), class_names, ignore_index)
        a3 = SegmentationMeter(len(class_names), class_names, ignore_index)
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"AppendixB({mode})", leave=False):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(
                        images, class_prompts=class_prompts, num_rounds=3,
                        top_m_windows=args.top_m_windows, routing_mode=mode,
                    )
                bmask = extract_boundary_mask(masks, ignore_index=ignore_index, radius=4)
                b_targets = torch.where(bmask, masks, torch.full_like(masks, ignore_index))
                u = torch.stack(out["active_masks_hi"], dim=0).amax(dim=0)
                u_up = F.interpolate(u, size=masks.shape[-2:], mode="nearest").squeeze(1) > 0.5
                act_targets = torch.where(u_up, masks, torch.full_like(masks, ignore_index))

                p1 = F.interpolate(out["round_logits"][0].float(), size=masks.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
                p3 = F.interpolate(out["round_logits"][2].float(), size=masks.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
                m1.update(p1, masks); m3.update(p3, masks)
                b1.update(p1, b_targets); b3.update(p3, b_targets)
                a1.update(p1, act_targets); a3.update(p3, act_targets)
        r1c, r3c = m1.compute(), m3.compute()
        appendix_b[mode] = {
            "R1_mIoU": r1c["mIoU"],
            "R3_mIoU": r3c["mIoU"],
            "delta_mIoU": round(r3c["mIoU"] - r1c["mIoU"], 2),
            "R1_bnd": b1.compute()["mIoU"],
            "R3_bnd": b3.compute()["mIoU"],
            "delta_bnd": round(b3.compute()["mIoU"] - b1.compute()["mIoU"], 2),
            "R1_active": a1.compute()["mIoU"],
            "R3_active": a3.compute()["mIoU"],
            "delta_active": round(a3.compute()["mIoU"] - a1.compute()["mIoU"], 2),
        }

    # Appendix C: LoveDA Urban vs Rural separate domain split (if LoveDA)
    appendix_c = {}
    if ds_cfg["name"].lower() == "loveda":
        for dom in ("Urban", "Rural"):
            dom_cfg = json.loads(json.dumps(cfg))
            dom_cfg["dataset"]["domains"] = [dom]
            _, dom_ds = build_rs_datasets(dom_cfg, cache_dir=args.cache_dir, max_train_samples=4, max_val_samples=200)
            dom_loader = DataLoader(dom_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
            dm1 = SegmentationMeter(len(class_names), class_names, ignore_index)
            dm3 = SegmentationMeter(len(class_names), class_names, ignore_index)
            db1 = SegmentationMeter(len(class_names), class_names, ignore_index)
            db3 = SegmentationMeter(len(class_names), class_names, ignore_index)
            with torch.inference_mode():
                for batch in tqdm(dom_loader, desc=f"AppendixC(LoveDA-{dom})", leave=False):
                    images = batch["image"].to(device, non_blocking=True)
                    masks = batch["mask"].to(device, non_blocking=True)
                    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                        out = model(images, class_prompts=class_prompts, num_rounds=3, top_m_windows=args.top_m_windows)
                    bmask = extract_boundary_mask(masks, ignore_index=ignore_index, radius=4)
                    b_targets = torch.where(bmask, masks, torch.full_like(masks, ignore_index))
                    p1 = F.interpolate(out["round_logits"][0].float(), size=masks.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
                    p3 = F.interpolate(out["round_logits"][2].float(), size=masks.shape[-2:], mode="bilinear", align_corners=False).argmax(dim=1)
                    dm1.update(p1, masks); dm3.update(p3, masks)
                    db1.update(p1, b_targets); db3.update(p3, b_targets)
            appendix_c[dom] = {
                "num_images": len(dom_ds),
                "R1": dm1.compute(),
                "R3": dm3.compute(),
                "R1_bnd": db1.compute()["mIoU"],
                "R3_bnd": db3.compute()["mIoU"],
            }

    report = {
        "dataset": ds_cfg["name"],
        "links_ckpt": args.links_ckpt,
        "bidir_sat_recursion": bool(args.bidir_sat_recursion),
        "appendix_a_depth_extrapolation_R1_to_R5": appendix_a,
        "appendix_b_routing_strategy_ablation": appendix_b,
        "appendix_c_domain_breakdown": appendix_c,
        "appendix_d_module_params_and_vram": {**module_params, "peak_eval_vram_gb": peak_vram_gb},
        "appendix_e_correction_panels": vis_stats,
        "appendix_f_round_embedding_alignment": appendix_f,
    }
    out_json = out_dir / f"{ds_cfg['name']}_appendix_a_to_e.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[appendix] Saved complete Appendix A-E report to {out_json}", flush=True)


if __name__ == "__main__":
    main()
