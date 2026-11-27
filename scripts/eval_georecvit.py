#!/usr/bin/env python3
"""Comprehensive Evaluation & Test-Time Scaling Sweep for GeoRecViT.

Measures:
  1. Test-Time Recurrent Depth Scaling (`R = 1, 2, 3, ...`)
  2. Sparse Window Budget Pareto Curve (`M = 1, 3, 5, 9` out of 9 windows)
  3. Per-class IoU, mIoU, mF1, OA, and GPU wall-clock latency (ms/image).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from georecvit.data import build_rs_datasets, load_yaml_config
from georecvit.models import DINOv3Expert, GeoRecViT, GeoRecViTConfig, SAM3RecursiveBackbone
from georecvit.utils import SegmentationMeter, extract_boundary_mask, load_georecvit_links


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate GeoRecViT test-time scaling & sparse routing.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--sam3_ckpt", type=str, required=True)
    p.add_argument("--dinov3_dir", type=str, default="")
    p.add_argument("--links_ckpt", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/content/georecvit_cache/datasets")
    p.add_argument("--rounds_list", type=int, nargs="+", default=[1, 2, 3])
    p.add_argument("--windows_list", type=int, nargs="+", default=[1, 3, 5, 9])
    p.add_argument("--no_dinov3", action="store_true")
    p.add_argument("--bidir_sat_recursion", action="store_true")
    p.add_argument("--dinov3_split_layer", type=int, default=18)
    p.add_argument("--detr_every_round", action="store_true")
    p.add_argument("--class_chunk_size", type=int, default=2)
    p.add_argument("--max_val_samples", type=int, default=-1)
    p.add_argument("--tta", action="store_true", help="Enable D4 horizontal/vertical flip TTA")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--output_json", type=str, default="results/eval_georecvit_sweep.json")
    return p.parse_args()


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
        class_chunk_size=args.class_chunk_size,
    )
    use_dinov3 = (not args.no_dinov3) and bool(m_cfg.get("use_dinov3", True)) and bool(args.dinov3_dir)
    dinov3 = (
        DINOv3Expert(
            model_path=args.dinov3_dir,
            input_size=int(ds_cfg.get("dinov3_resolution", 512)),
            split_layer_idx=int(args.dinov3_split_layer),
            use_grad_checkpoint=False,
            dtype=amp_dtype if use_amp else torch.float32,
        )
        if use_dinov3
        else None
    )
    model_cfg = GeoRecViTConfig(
        num_classes=int(ds_cfg["num_classes"]),
        pe_dim=sam3_wrapper.embed_dim,
        sat_dim=dinov3.embed_dim if dinov3 is not None else 1024,
        decoder_dim=sam3_wrapper.d_model,
        window_size=sam3_wrapper.window_size,
        num_recursive_rounds=max(args.rounds_list),
        sparse_top_windows=int(m_cfg.get("sparse_top_windows", 3)),
        use_sam3_detr=bool(m_cfg.get("use_sam3_detr", True)),
        detr_round1_only=(not args.detr_every_round) and bool(m_cfg.get("detr_round1_only", True)),
        use_dinov3=use_dinov3,
        use_inner_link=True,
        use_feedback_loop=(max(args.rounds_list) > 1),
        bidir_sat_recursion=bool(args.bidir_sat_recursion),
    )
    model = GeoRecViT(sam3_wrapper, dinov3, model_cfg).to(device)
    load_georecvit_links(model, args.links_ckpt, strict=False)
    model.eval()

    _, val_ds = build_rs_datasets(
        cfg,
        cache_dir=args.cache_dir,
        max_train_samples=4,
        max_val_samples=(args.max_val_samples if args.max_val_samples > 0 else None),
    )
    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    class_prompts = ds_cfg["class_prompts"]
    class_names = ds_cfg["class_names"]
    ignore_index = int(ds_cfg.get("ignore_index", 255))

    results = {"dataset": ds_cfg["name"], "params": model.count_parameters(), "sweeps": []}
    max_r = max(args.rounds_list)

    for top_m in args.windows_list:
        meters = {r: SegmentationMeter(len(class_names), class_names, ignore_index) for r in args.rounds_list}
        boundary_meters = {r: SegmentationMeter(len(class_names), class_names, ignore_index) for r in args.rounds_list}
        active_meters = {r: SegmentationMeter(len(class_names), class_names, ignore_index) for r in args.rounds_list}
        t_total = 0.0
        n_img = 0

        with torch.inference_mode():
            for batch in tqdm(loader, desc=f"Sweep(M={top_m}/9,R_max={max_r})"):
                images = batch["image"].to(device, non_blocking=True)
                masks = batch["mask"].to(device, non_blocking=True)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                    out = model(
                        images,
                        class_prompts=class_prompts,
                        num_rounds=max_r,
                        top_m_windows=top_m,
                    )
                    if args.tta:
                        out_h = model(
                            torch.flip(images, dims=[-1]),
                            class_prompts=class_prompts,
                            num_rounds=max_r,
                            top_m_windows=top_m,
                        )
                        out_v = model(
                            torch.flip(images, dims=[-2]),
                            class_prompts=class_prompts,
                            num_rounds=max_r,
                            top_m_windows=top_m,
                        )
                        out["round_logits"] = [
                            (
                                out["round_logits"][idx].float()
                                + torch.flip(out_h["round_logits"][idx].float(), dims=[-1])
                                + torch.flip(out_v["round_logits"][idx].float(), dims=[-2])
                            )
                            / 3.0
                            for idx in range(len(out["round_logits"]))
                        ]
                if device.type == "cuda":
                    torch.cuda.synchronize()
                t_total += time.perf_counter() - t0
                n_img += images.shape[0]

                bmask = extract_boundary_mask(masks, ignore_index=ignore_index, radius=4)
                boundary_targets = torch.where(bmask, masks, torch.full_like(masks, ignore_index))

                union_active = None
                if out.get("active_masks_hi"):
                    u = torch.stack(out["active_masks_hi"], dim=0).amax(dim=0)
                    u_up = F.interpolate(u, size=masks.shape[-2:], mode="nearest").squeeze(1) > 0.5
                    union_active = torch.where(u_up, masks, torch.full_like(masks, ignore_index))

                for r in args.rounds_list:
                    lg = out["round_logits"][r - 1]
                    lg_up = F.interpolate(
                        lg.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False
                    )
                    pred = lg_up.argmax(dim=1)
                    meters[r].update(pred, masks)
                    boundary_meters[r].update(pred, boundary_targets)
                    if union_active is not None:
                        active_meters[r].update(pred, union_active)

        entry = {
            "top_m_windows": top_m,
            "active_window_ratio": round(top_m / 9.0, 3),
            "avg_latency_ms_at_max_R": round(1000.0 * t_total / max(n_img, 1), 2),
            "by_round": {f"R={r}": meters[r].compute() for r in args.rounds_list},
            "boundary_by_round_mIoU": {f"R={r}": boundary_meters[r].compute()["mIoU"] for r in args.rounds_list},
            "active_window_by_round_mIoU": {f"R={r}": active_meters[r].compute()["mIoU"] for r in args.rounds_list},
        }
        results["sweeps"].append(entry)
        print(json.dumps(entry, indent=2), flush=True)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"[eval] Saved sweep results to {out_path}", flush=True)


if __name__ == "__main__":
    main()
