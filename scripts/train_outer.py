#!/usr/bin/env python3
"""Stage 2: Outer-Loop Closed-Loop Multi-Round Co-Optimization (cf. RecursiveMAS `train_outer.py`).

Supports controlled experiment modes via CLI flags so all baselines share identical splits:
  - Baseline 1 (`--no_dinov3 --num_rounds 1`): Frozen SAM 3 + single-pass readout (no SAT, no recursion)
  - Baseline 2 (`--num_rounds 1`): Frozen SAM 3 + DINOv3-SAT single-pass (R=1, no feedback recursion)
  - GeoRecViT  (`--num_rounds 3 --top_m_windows 3`): Full closed-loop sparse ViT recursion!
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from georecvit.data import build_rs_datasets, load_yaml_config
from georecvit.losses import MultiRoundSegmentationLoss
from georecvit.models import DINOv3Expert, GeoRecViT, GeoRecViTConfig, SAM3RecursiveBackbone
from georecvit.utils import SegmentationMeter, load_georecvit_links, save_georecvit_links


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 2: Outer-Loop multi-round training for GeoRecViT.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--sam3_ckpt", type=str, required=True)
    p.add_argument("--dinov3_dir", type=str, default="")
    p.add_argument("--inner_ckpt", type=str, default="", help="Optional Stage 1 inner warm-start .pt")
    p.add_argument("--anchor_ckpt", type=str, default="", help="Optional Stage 2a single-pass anchor .pt")
    p.add_argument("--freeze_round1_anchor", action="store_true", help="Stage 2b: freeze Round-1 anchor and train ONLY closed-loop recursion triad")
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--active_refine_weight", type=float, default=0.5)
    p.add_argument("--cache_dir", type=str, default="/content/georecvit_cache/datasets")
    p.add_argument("--save_dir", type=str, default="/content/georecvit_cache/checkpoints/outer")
    p.add_argument("--num_rounds", type=int, default=None)
    p.add_argument("--top_m_windows", type=int, default=None)
    p.add_argument("--no_dinov3", action="store_true")
    p.add_argument("--no_detr", action="store_true")
    p.add_argument("--detr_every_round", action="store_true", help="Run full SAM 3 DETR on every round (slower)")
    p.add_argument("--no_inner_link", action="store_true")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--grad_accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_val_samples", type=int, default=-1)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--class_chunk_size", type=int, default=2, help="Classes per DETR chunk to cap VRAM")
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed for fair controlled baselines")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: GeoRecViT,
    val_loader: DataLoader,
    class_prompts: list[str],
    class_names: list[str],
    ignore_index: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    use_amp: bool,
    num_rounds: int,
    top_m_windows: int,
) -> dict:
    model.eval()
    meters = [SegmentationMeter(len(class_names), class_names, ignore_index) for _ in range(num_rounds)]
    active_meters = (
        [SegmentationMeter(len(class_names), class_names, ignore_index) for _ in range(num_rounds)]
        if num_rounds > 1
        else None
    )
    t_total = 0.0
    n_img = 0
    for batch in tqdm(val_loader, desc=f"Val(R={num_rounds},M={top_m_windows})", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(
                images,
                class_prompts=class_prompts,
                num_rounds=num_rounds,
                top_m_windows=top_m_windows,
            )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        n_img += images.shape[0]

        union_active = None
        if active_meters is not None and out.get("active_masks_hi"):
            u = torch.stack(out["active_masks_hi"], dim=0).amax(dim=0)  # [B, 1, 288, 288]
            u_up = F.interpolate(u, size=masks.shape[-2:], mode="nearest").squeeze(1) > 0.5
            union_active = torch.where(u_up, masks, torch.full_like(masks, ignore_index))

        for r, lg in enumerate(out["round_logits"]):
            lg_up = F.interpolate(lg.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = lg_up.argmax(dim=1)
            meters[r].update(pred, masks)
            if active_meters is not None and union_active is not None:
                active_meters[r].update(pred, union_active)

    per_round = [m.compute() for m in meters]
    active_per_round_miou = (
        [m.compute()["mIoU"] for m in active_meters] if active_meters is not None else []
    )
    return {
        "final": per_round[-1],
        "per_round_mIoU": [r["mIoU"] for r in per_round],
        "active_window_per_round_mIoU": active_per_round_miou,
        "avg_val_latency_ms": round(1000.0 * t_total / max(n_img, 1), 2),
        "per_round": per_round,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_yaml_config(args.config)
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    t_cfg = cfg["train_outer"]

    num_rounds = args.num_rounds if args.num_rounds is not None else int(m_cfg.get("num_recursive_rounds", 3))
    top_m = args.top_m_windows if args.top_m_windows is not None else int(m_cfg.get("sparse_top_windows", 3))
    epochs = args.epochs or int(t_cfg["epochs"])
    batch_size = args.batch_size or int(t_cfg["batch_size"])
    grad_accum = args.grad_accum or int(t_cfg.get("grad_accum_steps", 8))
    lr = args.lr or float(t_cfg["lr"])
    use_dinov3 = (not args.no_dinov3) and bool(m_cfg.get("use_dinov3", True)) and bool(args.dinov3_dir)

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
        # Cast heavy frozen backbone Linear/Conv/Embedding weights to fp16/bf16 (-1.5 GB VRAM)
        # while keeping norms and DETR decoder (which disables autocast in FFN) in fp32.
        for mod in sam3_model.backbone.modules():
            if isinstance(mod, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.ConvTranspose2d, torch.nn.Embedding)):
                mod.to(dtype=amp_dtype)
    sam3_wrapper = SAM3RecursiveBackbone(
        sam3_model=sam3_model,
        split_block_idx=int(m_cfg.get("split_block_idx", 24)),
        resolution=int(ds_cfg.get("sam3_resolution", 1008)),
        use_grad_checkpoint=True,
        class_chunk_size=args.class_chunk_size,
    )
    dinov3 = (
        DINOv3Expert(
            model_path=args.dinov3_dir,
            input_size=int(ds_cfg.get("dinov3_resolution", 512)),
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
        num_recursive_rounds=num_rounds,
        sparse_top_windows=top_m,
        use_sam3_detr=(not args.no_detr) and bool(m_cfg.get("use_sam3_detr", True)),
        detr_round1_only=(not args.detr_every_round) and bool(m_cfg.get("detr_round1_only", True)),
        use_dinov3=use_dinov3,
        use_inner_link=(not args.no_inner_link) and bool(m_cfg.get("use_inner_link", True)),
        use_feedback_loop=(num_rounds > 1) and bool(m_cfg.get("use_feedback_loop", True)),
    )
    model = GeoRecViT(sam3_wrapper, dinov3, model_cfg).to(device)
    if args.inner_ckpt:
        loaded = load_georecvit_links(model, args.inner_ckpt, strict=False)
        print(f"[outer] Loaded inner warm-start from {args.inner_ckpt}: {list(loaded.keys())}", flush=True)
    if args.anchor_ckpt:
        loaded = load_georecvit_links(model, args.anchor_ckpt, strict=False)
        print(f"[outer] Loaded Stage 2a anchor from {args.anchor_ckpt}: {list(loaded.keys())}", flush=True)
    if args.freeze_round1_anchor:
        model.freeze_round1_anchor()
        print("[outer] Froze Round-1 anchor (sat_outer_link, inner_link, readout) for Stage 2b recursion!", flush=True)

    print(f"[outer] Model config: {model_cfg}", flush=True)
    print(f"[outer] Params: {model.count_parameters()}", flush=True)

    train_ds, val_ds = build_rs_datasets(
        cfg,
        cache_dir=args.cache_dir,
        max_train_samples=(args.max_train_samples if args.max_train_samples > 0 else None),
        max_val_samples=(args.max_val_samples if args.max_val_samples > 0 else None),
    )
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    criterion = MultiRoundSegmentationLoss(
        ignore_index=int(ds_cfg.get("ignore_index", 255)),
        non_last_loss_weight=float(t_cfg.get("non_last_loss_weight", 0.2)),
        ce_weight=float(t_cfg.get("ce_weight", 1.0)),
        dice_weight=float(t_cfg.get("dice_weight", 1.0)),
        focal_weight=float(t_cfg.get("focal_weight", 0.5)),
        active_refine_weight=float(args.active_refine_weight),
    )
    params = model.trainable_parameters()
    wd = args.weight_decay if args.weight_decay is not None else float(t_cfg.get("weight_decay", 0.01))
    optimizer = torch.optim.AdamW(
        params, lr=lr, weight_decay=wd, betas=(0.9, 0.95)
    )
    total_opt_steps = max(1, (epochs * len(train_loader)) // grad_accum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"))

    class_prompts = ds_cfg["class_prompts"]
    class_names = ds_cfg["class_names"]
    ignore_index = int(ds_cfg.get("ignore_index", 255))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    best_miou = -1.0
    history = []

    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        steps = 0

        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"Outer-Ep {ep}/{epochs}"), start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)

            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(
                    images,
                    class_prompts=class_prompts,
                    num_rounds=num_rounds,
                    top_m_windows=top_m,
                )
                losses = criterion(
                    out["round_logits"],
                    masks,
                    active_masks_hi=out.get("active_masks_hi"),
                )
                loss = losses["loss"] / grad_accum

            scaler.scale(loss).backward()
            loss_acc += float(losses["loss"].item())
            steps += 1

            if step_idx % grad_accum == 0 or step_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_res = evaluate(
            model, val_loader, class_prompts, class_names, ignore_index,
            device, amp_dtype, use_amp, num_rounds, top_m
        )
        cur_miou = float(val_res["final"]["mIoU"])
        rec = {
            "epoch": ep,
            "train_loss": round(loss_acc / max(steps, 1), 4),
            "val_mIoU": cur_miou,
            "val_per_round_mIoU": val_res["per_round_mIoU"],
            "active_window_per_round_mIoU": val_res["active_window_per_round_mIoU"],
            "avg_val_latency_ms": val_res["avg_val_latency_ms"],
            "val_mF1": val_res["final"]["mF1"],
            "val_OA": val_res["final"]["OA"],
            "per_class_IoU": val_res["final"]["per_class_IoU"],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"[outer] ep={ep}/{epochs} loss={rec['train_loss']:.4f} "
            f"val_mIoU={cur_miou:.2f} rounds={rec['val_per_round_mIoU']} "
            f"active_win={rec['active_window_per_round_mIoU']} ms={rec['avg_val_latency_ms']}",
            flush=True,
        )

        if cur_miou > best_miou:
            best_miou = cur_miou
            save_georecvit_links(model, save_dir, tag="best", extra_meta=rec)
        save_georecvit_links(model, save_dir, tag="last", extra_meta=rec)
        with open(save_dir / "train_history.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"[outer] Done! Best val mIoU = {best_miou:.2f}%", flush=True)


if __name__ == "__main__":
    main()
