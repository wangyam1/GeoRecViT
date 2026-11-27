#!/usr/bin/env python3
"""Stage 1: Inner-Loop Manifold Alignment Warm-Start (cf. RecursiveMAS `train_inner.py`).

Freezes BOTH SAM 3 PE-ViT and DINOv3-SAT-493M, and trains ONLY `sat_outer_link` +
`inner_link` with Cosine + MSE manifold alignment loss so heterogeneous satellite
features enter SAM 3's frozen `blocks[24:32]` strictly in-distribution.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from georecvit.data import build_rs_datasets, load_yaml_config
from georecvit.models import DINOv3Expert, GeoRecViT, GeoRecViTConfig, SAM3RecursiveBackbone
from georecvit.utils import save_georecvit_links


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Stage 1: Inner-Loop manifold warm-start for GeoRecViT.")
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--sam3_ckpt", type=str, required=True)
    p.add_argument("--dinov3_dir", type=str, required=True)
    p.add_argument("--cache_dir", type=str, default="/content/georecvit_cache/datasets")
    p.add_argument("--save_dir", type=str, default="/content/georecvit_cache/checkpoints/inner")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max_samples", type=int, default=None)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml_config(args.config)
    ds_cfg = cfg["dataset"]
    m_cfg = cfg["model"]
    t_cfg = cfg["train_inner"]

    epochs = args.epochs or int(t_cfg["epochs"])
    batch_size = args.batch_size or int(t_cfg["batch_size"])
    lr = args.lr or float(t_cfg["lr"])
    max_samples = args.max_samples if args.max_samples is not None else int(t_cfg.get("max_samples", 800))

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
    sam3_wrapper = SAM3RecursiveBackbone(
        sam3_model=sam3_model,
        split_block_idx=int(m_cfg.get("split_block_idx", 24)),
        resolution=int(ds_cfg.get("sam3_resolution", 1008)),
        use_grad_checkpoint=False,
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
        num_recursive_rounds=int(m_cfg.get("num_recursive_rounds", 3)),
        sparse_top_windows=int(m_cfg.get("sparse_top_windows", 3)),
        use_sam3_detr=bool(m_cfg.get("use_sam3_detr", True)),
        use_dinov3=True,
        use_inner_link=True,
        use_feedback_loop=True,
    )
    model = GeoRecViT(sam3_wrapper, dinov3, model_cfg).to(device)
    model.train()

    inner_params = []
    if model.sat_outer_link is not None:
        inner_params.extend(model.sat_outer_link.parameters())
    if model.inner_link is not None:
        inner_params.extend(model.inner_link.parameters())

    print(f"[inner] Param summary: {model.count_parameters()}", flush=True)
    optimizer = torch.optim.AdamW(
        inner_params, lr=lr, weight_decay=float(t_cfg.get("weight_decay", 0.01)), betas=(0.9, 0.95)
    )

    train_ds, _ = build_rs_datasets(
        cfg, cache_dir=args.cache_dir, max_train_samples=max_samples, max_val_samples=4
    )
    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs * len(loader)))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.precision == "fp16" and device.type == "cuda"))

    t0 = time.time()
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        for batch in tqdm(loader, desc=f"Inner-Epoch {ep}/{epochs}"):
            images = batch["image"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = model.forward_inner_alignment(
                    images,
                    cos_weight=float(t_cfg.get("cos_weight", 1.0)),
                    mse_weight=float(t_cfg.get("mse_weight", 0.1)),
                )
                loss = out["loss"]
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(inner_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += float(loss.item())

        avg_loss = loss_sum / max(len(loader), 1)
        print(
            f"[inner] epoch={ep}/{epochs} avg_loss={avg_loss:.4f} elapsed={time.time()-t0:.1f}s",
            flush=True,
        )

    ckpt_path = save_georecvit_links(
        model,
        output_dir=args.save_dir,
        tag="inner_warmstart",
        extra_meta={"dataset": ds_cfg["name"], "epochs": epochs, "final_loss": avg_loss},
    )
    print(f"[inner] Saved inner warm-start checkpoint to {ckpt_path}", flush=True)


if __name__ == "__main__":
    main()
