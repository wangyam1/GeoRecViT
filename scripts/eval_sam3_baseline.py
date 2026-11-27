#!/usr/bin/env python3
"""Baseline 0: Zero-Shot Frozen SAM 3 Evaluation on LoveDA / OpenEarthMap.

Reproduces two apples-to-apples SAM 3 baselines before any training:
  1. `sam3_semantic_only`: argmax over SAM 3's pretrained `semantic_seg` head
  2. `sam3_dual_head` (SegEarth-OV3-style): fusion of `semantic_seg` + presence-gated
     instance projection (`inst_probs`)
Outputs JSON report with mIoU, mF1, OA, per-class IoU, and average latency/image.
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
from georecvit.models import SAM3RecursiveBackbone
from georecvit.utils import SegmentationMeter


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate zero-shot SAM 3 baseline on RS benchmarks.")
    p.add_argument("--config", type=str, required=True, help="Path to YAML config (loveda.yaml / openearthmap.yaml)")
    p.add_argument("--sam3_ckpt", type=str, required=True, help="Path to sam3.pt downloaded from ModelScope")
    p.add_argument("--cache_dir", type=str, default="/content/georecvit_cache/datasets")
    p.add_argument("--max_val_samples", type=int, default=-1, help="Limit val samples for quick smoke test (-1 = full)")
    p.add_argument("--dual_alpha", type=float, default=0.5, help="Weight on instance-presence map in dual-head fusion")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--output_json", type=str, default="results/baseline_sam3_zeroshot.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    ds_cfg = cfg["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    use_amp = device.type == "cuda" and args.precision != "fp32"

    print(f"[baseline0] Loading SAM 3 from {args.sam3_ckpt}...", flush=True)
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
    wrapper = SAM3RecursiveBackbone(
        sam3_model=sam3_model,
        split_block_idx=int(cfg["model"].get("split_block_idx", 24)),
        resolution=int(ds_cfg.get("sam3_resolution", 1008)),
        use_grad_checkpoint=False,
    ).to(device)
    wrapper.eval()

    max_val = args.max_val_samples if args.max_val_samples > 0 else None
    _, val_ds = build_rs_datasets(
        cfg,
        cache_dir=args.cache_dir,
        max_train_samples=4,  # tiny train handle, unused
        max_val_samples=max_val,
    )
    print(
        f"[baseline0] Dataset={ds_cfg['name']}, val_len={len(val_ds)}, "
        f"detected_label_offset={val_ds.label_offset}",
        flush=True,
    )

    loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    class_prompts = ds_cfg["class_prompts"]
    class_names = ds_cfg["class_names"]
    num_classes = int(ds_cfg["num_classes"])
    ignore_index = int(ds_cfg.get("ignore_index", 255))

    meter_sem = SegmentationMeter(num_classes, class_names, ignore_index)
    meter_dual = SegmentationMeter(num_classes, class_names, ignore_index)

    text_out = wrapper.encode_class_prompts(class_prompts, device=device)

    total_time = 0.0
    num_images = 0

    with torch.inference_mode():
        for batch in tqdm(loader, desc=f"SAM3-ZeroShot-{ds_cfg['name']}"):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                f_mid = wrapper.forward_early_vit(images)
                late = wrapper.forward_late_vit(f_mid)
                fpn, pos = wrapper.forward_neck(late.vit_out)
                dec = wrapper.forward_concept_decode(fpn, pos, text_out, run_detr=True)

            if device.type == "cuda":
                torch.cuda.synchronize()
            total_time += time.perf_counter() - t0
            num_images += images.shape[0]

            # Semantic head prediction
            sem_up = F.interpolate(
                dec.sem_logits.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            pred_sem = sem_up.argmax(dim=1)
            meter_sem.update(pred_sem, masks)

            # Dual-head (SegEarth-OV3 style): combine semantic prob + presence-gated instance prob
            sem_prob = sem_up.sigmoid() * dec.presence_probs[:, :, None, None].float()
            inst_up = F.interpolate(
                dec.inst_probs.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False
            )
            dual_score = (1.0 - args.dual_alpha) * sem_prob + args.dual_alpha * inst_up
            pred_dual = dual_score.argmax(dim=1)
            meter_dual.update(pred_dual, masks)

    res_sem = meter_sem.compute()
    res_dual = meter_dual.compute()
    avg_ms = round(1000.0 * total_time / max(num_images, 1), 2)

    report = {
        "dataset": ds_cfg["name"],
        "num_val_images": num_images,
        "label_offset": val_ds.label_offset,
        "avg_latency_ms": avg_ms,
        "class_prompts": class_prompts,
        "sam3_semantic_only": res_sem,
        "sam3_dual_head": res_dual,
    }
    print(json.dumps(report, indent=2), flush=True)

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"[baseline0] Saved report to {out_path}", flush=True)


if __name__ == "__main__":
    main()
