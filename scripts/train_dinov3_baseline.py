#!/usr/bin/env python3
"""Controlled Baseline 1b (`B` Single-Pass) & `B <-> B` Self-Recursion for Frozen DINOv3-SAT.

Supports:
  - `--num_rounds 1` (`B` only): Frozen DINOv3-SAT + ASPP progressive 288x288 decoder (~3.74M params)
  - `--num_rounds 3 --anchor_ckpt ... --freeze_round1_anchor` (`B <-> B` self-recursion):
    Re-enters frozen DINOv3-SAT late blocks `layers[18:24]` with entropy-routed sparse feedback + CSPR!
"""

from __future__ import annotations

import argparse
import json
import math
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
from georecvit.models import DINOv3Expert, DINOv3OnlySegmenter
from georecvit.utils import SegmentationMeter, extract_boundary_mask


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Baseline 1b (B) and B<->B Self-Recursion on Frozen DINOv3-SAT.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--dinov3_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/content/georecvit_cache/datasets")
    p.add_argument("--save_dir", type=str, default="/content/georecvit_cache/checkpoints/dinov3_only")
    p.add_argument("--output_json", type=str, default="")
    p.add_argument("--anchor_ckpt", type=str, default="", help="Optional Round-1 DINOv3-Only ckpt for Stage 2b B<->B")
    p.add_argument("--freeze_round1_anchor", action="store_true")
    p.add_argument("--anchor_perturb_prob", type=float, default=0.0)
    p.add_argument("--num_rounds", type=int, default=1)
    p.add_argument("--top_m_windows", type=int, default=3)
    p.add_argument("--dinov3_split_layer", type=int, default=18)
    p.add_argument("--active_refine_weight", type=float, default=1.5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--max_train_samples", type=int, default=-1)
    p.add_argument("--max_val_samples", type=int, default=-1)
    p.add_argument("--eval_full_val", action="store_true")
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate(
    model: DINOv3OnlySegmenter,
    val_loader: DataLoader,
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
    boundary_meters = [SegmentationMeter(len(class_names), class_names, ignore_index) for _ in range(num_rounds)]
    active_meters = (
        [SegmentationMeter(len(class_names), class_names, ignore_index) for _ in range(num_rounds)]
        if num_rounds > 1
        else None
    )
    t_total = 0.0
    n_img = 0
    for batch in tqdm(val_loader, desc=f"Val(DINOv3 R={num_rounds},M={top_m_windows})", leave=False):
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(images, num_rounds=num_rounds, top_m_windows=top_m_windows)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_total += time.perf_counter() - t0
        n_img += images.shape[0]

        bmask = extract_boundary_mask(masks, ignore_index=ignore_index, radius=4)
        b_targets = torch.where(bmask, masks, torch.full_like(masks, ignore_index))

        union_active = None
        if active_meters is not None and out.get("active_masks_hi"):
            u = torch.stack(out["active_masks_hi"], dim=0).amax(dim=0)
            u_up = F.interpolate(u, size=masks.shape[-2:], mode="nearest").squeeze(1) > 0.5
            union_active = torch.where(u_up, masks, torch.full_like(masks, ignore_index))

        for r, lg in enumerate(out["round_logits"]):
            lg_up = F.interpolate(lg.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
            pred = lg_up.argmax(dim=1)
            meters[r].update(pred, masks)
            boundary_meters[r].update(pred, b_targets)
            if active_meters is not None and union_active is not None:
                active_meters[r].update(pred, union_active)

    per_round = [m.compute() for m in meters]
    return {
        "final": per_round[-1],
        "per_round_mIoU": [r["mIoU"] for r in per_round],
        "boundary_per_round_mIoU": [m.compute()["mIoU"] for m in boundary_meters],
        "active_window_per_round_mIoU": (
            [m.compute()["mIoU"] for m in active_meters] if active_meters is not None else []
        ),
        "avg_val_latency_ms": round(1000.0 * t_total / max(n_img, 1), 2),
        "per_round": per_round,
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    cfg = load_yaml_config(args.config)
    ds_cfg = cfg["dataset"]
    t_cfg = cfg["train_outer"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]
    use_amp = device.type == "cuda" and args.precision != "fp32"

    dinov3 = DINOv3Expert(
        model_path=args.dinov3_dir,
        input_size=int(ds_cfg.get("dinov3_resolution", 512)),
        split_layer_idx=int(args.dinov3_split_layer),
        use_grad_checkpoint=True,
        dtype=amp_dtype if use_amp else torch.float32,
    )
    model = DINOv3OnlySegmenter(
        dinov3_expert=dinov3,
        num_classes=int(ds_cfg["num_classes"]),
        hidden_dim=args.hidden_dim,
        out_size=288,
        num_rounds=args.num_rounds,
        top_m_windows=args.top_m_windows,
        anchor_perturb_prob=args.anchor_perturb_prob,
    ).to(device)
    if args.anchor_ckpt:
        ckpt = torch.load(args.anchor_ckpt, map_location="cpu")
        state = ckpt.get("decoder", ckpt)
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[dinov3_only] Loaded anchor {args.anchor_ckpt} (missing={len(missing)}, unexpected={len(unexpected)})", flush=True)
    if args.freeze_round1_anchor:
        model.freeze_round1_anchor()
        print("[dinov3_only] Froze Round-1 decoder for decoupled B<->B self-recursion!", flush=True)

    param_info = model.count_parameters()
    print(f"[dinov3_only] R={args.num_rounds}, M={args.top_m_windows}, Params: {param_info}", flush=True)

    train_ds, val_ds = build_rs_datasets(
        cfg,
        cache_dir=args.cache_dir,
        max_train_samples=(args.max_train_samples if args.max_train_samples > 0 else None),
        max_val_samples=(args.max_val_samples if args.max_val_samples > 0 else None),
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

    criterion = MultiRoundSegmentationLoss(
        ignore_index=int(ds_cfg.get("ignore_index", 255)),
        non_last_loss_weight=(0.25 if args.num_rounds > 1 else 0.0),
        ce_weight=float(t_cfg.get("ce_weight", 1.0)),
        dice_weight=float(t_cfg.get("dice_weight", 1.0)),
        focal_weight=float(t_cfg.get("focal_weight", 0.5)),
        active_refine_weight=(args.active_refine_weight if args.num_rounds > 1 else 0.0),
        active_dice_weight=1.0,
        boundary_weight=0.5,
        skip_round1_loss=bool(args.freeze_round1_anchor),
    )
    params = model.trainable_parameters()
    decay_params = [p for p in params if p.ndim >= 2]
    no_decay_params = [p for p in params if p.ndim < 2]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": args.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.95),
    )
    total_opt_steps = max(1, (args.epochs * len(train_loader)) // args.grad_accum)
    warmup_steps = max(1, int(total_opt_steps * float(t_cfg.get("warmup_ratio", 0.08))))

    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        prog = float(step - warmup_steps) / float(max(1, total_opt_steps - warmup_steps))
        return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"))

    class_names = ds_cfg["class_names"]
    ignore_index = int(ds_cfg.get("ignore_index", 255))
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    best_miou = -1.0
    best_state = None
    history = []
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_acc = 0.0
        steps = 0
        for step_idx, batch in enumerate(tqdm(train_loader, desc=f"DINOv3(R={args.num_rounds})-Ep {ep}/{args.epochs}"), start=1):
            images = batch["image"].to(device, non_blocking=True)
            masks = batch["mask"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model(images, num_rounds=args.num_rounds, top_m_windows=args.top_m_windows)
                losses = criterion(out["round_logits"], masks, active_masks_hi=out.get("active_masks_hi"))
                loss = losses["loss"] / args.grad_accum
            scaler.scale(loss).backward()
            loss_acc += float(losses["loss"].item())
            steps += 1
            if step_idx % args.grad_accum == 0 or step_idx == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

        val_res = evaluate(
            model, val_loader, class_names, ignore_index, device, amp_dtype, use_amp,
            args.num_rounds, args.top_m_windows,
        )
        cur_miou = float(val_res["final"]["mIoU"])
        rec = {
            "epoch": ep,
            "train_loss": round(loss_acc / max(steps, 1), 4),
            "val_mIoU": cur_miou,
            "val_per_round_mIoU": val_res["per_round_mIoU"],
            "boundary_per_round_mIoU": val_res["boundary_per_round_mIoU"],
            "active_window_per_round_mIoU": val_res["active_window_per_round_mIoU"],
            "avg_val_latency_ms": val_res["avg_val_latency_ms"],
            "val_mF1": val_res["final"]["mF1"],
            "val_OA": val_res["final"]["OA"],
            "per_class_IoU": val_res["final"]["per_class_IoU"],
            "elapsed_sec": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(
            f"[dinov3_only] ep={ep}/{args.epochs} loss={rec['train_loss']:.4f} "
            f"val_mIoU={cur_miou:.2f} rounds={rec['val_per_round_mIoU']} "
            f"bnd={rec['boundary_per_round_mIoU']} active={rec['active_window_per_round_mIoU']} "
            f"ms={rec['avg_val_latency_ms']}",
            flush=True,
        )
        if cur_miou > best_miou:
            best_miou = cur_miou
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
                if not k.startswith("dinov3_expert.")
            }
            torch.save({"decoder": best_state, "meta": rec, "params": param_info}, save_dir / "dinov3_only_best.pt")

    full_val_res = None
    if args.eval_full_val and best_state is not None:
        model.load_state_dict(best_state, strict=False)
        _, full_val_ds = build_rs_datasets(cfg, cache_dir=args.cache_dir, max_train_samples=4, max_val_samples=None)
        full_loader = DataLoader(full_val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
        full_val_res = evaluate(
            model, full_loader, class_names, ignore_index, device, amp_dtype, use_amp,
            args.num_rounds, args.top_m_windows,
        )
        print(
            f"[dinov3_only] FULL VAL ({len(full_val_ds)} imgs): "
            f"rounds={full_val_res['per_round_mIoU']} bnd={full_val_res['boundary_per_round_mIoU']} "
            f"mF1={full_val_res['final']['mF1']:.2f} OA={full_val_res['final']['OA']:.2f}",
            flush=True,
        )

    summary = {
        "dataset": ds_cfg["name"],
        "dinov3_dir": args.dinov3_dir,
        "num_rounds": args.num_rounds,
        "top_m_windows": args.top_m_windows,
        "params": param_info,
        "train_samples": len(train_ds),
        "val_samples": len(val_ds),
        "best_val_mIoU": best_miou,
        "history": history,
        "full_val": full_val_res,
    }
    with open(save_dir / "train_history.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[dinov3_only] Saved summary JSON to {out_p}", flush=True)


if __name__ == "__main__":
    main()
