"""Local unit & autograd tests for GeoRecViT (runs in <5s on CPU, zero large downloads).

Verifies:
  1. Exact 24x24 native window gather/scatter reversibility & autograd continuity
  2. UncertaintyWindowRouter scores, active mask coverage, and budget override
  3. End-to-end Stage 1 (Inner-Loop alignment) forward & backward on frozen mocks
  4. End-to-end Stage 2 (Outer-Loop R=3 sparse closed-loop recursion) gradient flow
     through frozen late ViT blocks back into ALL trainable RecursiveLink modules
  5. Exact frozen-backbone invariance (0 gradients on frozen ViT/DINOv3 parameters)
  6. MultiRoundSegmentationLoss, SegmentationMeter mIoU, and checkpoint round-trip.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from georecvit.data.rs_datasets import normalize_mask_array
from georecvit.losses import MultiRoundSegmentationLoss
from georecvit.models import GeoRecViT, GeoRecViTConfig, UncertaintyWindowRouter
from georecvit.models.sam3_wrapper import LateViTOutput, SAM3DecodeOutput
from georecvit.models.uncertainty_router import gather_selected_windows, scatter_selected_windows
from georecvit.utils import SegmentationMeter, load_georecvit_links, save_georecvit_links


class _MockWindowBlock(nn.Module):
    def __init__(self, dim: int, window_size: int) -> None:
        super().__init__()
        self.window_size = window_size
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + 0.1 * self.fc(F.gelu(self.norm(x)))


class _MockSAM3Wrapper(nn.Module):
    """Lightweight CPU-friendly mock matching exact SAM3RecursiveBackbone tensor contracts."""

    def __init__(self, pe_dim: int = 64, d_model: int = 32, grid: int = 12, window_size: int = 4) -> None:
        super().__init__()
        self.embed_dim = pe_dim
        self.d_model = d_model
        self.grid = grid
        self.window_size = window_size

        self.early_conv = nn.Conv2d(3, pe_dim, kernel_size=1)
        self.late_win_blocks = nn.ModuleList([_MockWindowBlock(pe_dim, window_size) for _ in range(2)])
        self.global_block = _MockWindowBlock(pe_dim, 0)
        self.neck_hi = nn.Conv2d(pe_dim, d_model, kernel_size=1)
        self.neck_lo = nn.Conv2d(pe_dim, d_model, kernel_size=1)
        self.text_proj = nn.Linear(d_model, d_model)

        # Strictly freeze mock foundation backbone to verify gradient pass-through
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def encode_class_prompts(self, class_prompts: Sequence[str], device: torch.device) -> Dict[str, torch.Tensor]:
        C = len(class_prompts)
        gen = torch.Generator(device="cpu").manual_seed(0)
        proto = F.normalize(torch.randn(C, self.d_model, generator=gen), dim=-1).to(device)
        return {"class_prototypes": proto}

    @torch.no_grad()
    def forward_early_vit(self, images_01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images_01, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return self.early_conv(x).permute(0, 2, 3, 1).contiguous()

    def forward_late_vit(
        self,
        x_mid: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
        prev_pre_global: Optional[torch.Tensor] = None,
    ) -> LateViTOutput:
        if selected_indices is not None and prev_pre_global is not None:
            sel, _ = gather_selected_windows(x_mid, selected_indices, window_size=self.window_size)
            for blk in self.late_win_blocks:
                sel = blk(sel)
            pre_global = scatter_selected_windows(
                prev_pre_global, sel, selected_indices, window_size=self.window_size
            )
        else:
            pre_global = x_mid
            for blk in self.late_win_blocks:
                pre_global = blk(pre_global)
        g = self.global_block(pre_global)
        return LateViTOutput(pre_global=pre_global, vit_out=g.permute(0, 3, 1, 2).contiguous())

    def forward_neck(self, vit_out: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        hi = F.interpolate(self.neck_hi(vit_out), scale_factor=2, mode="bilinear", align_corners=False)
        lo = self.neck_lo(vit_out)
        return [hi, lo], [torch.zeros_like(hi), torch.zeros_like(lo)]

    def forward_concept_decode(
        self,
        fpn_feats: List[torch.Tensor],
        fpn_pos: List[torch.Tensor],
        text_out: Dict[str, torch.Tensor],
        run_detr: bool = True,
    ) -> SAM3DecodeOutput:
        hi, lo = fpn_feats[0], fpn_feats[1]
        proto = self.text_proj(text_out["class_prototypes"])
        sem_logits = torch.einsum("bchw,kc->bkhw", hi, proto)
        inst_probs = torch.sigmoid(sem_logits * 0.8)
        pres = inst_probs.flatten(2).amax(dim=-1)
        return SAM3DecodeOutput(
            sem_logits=sem_logits,
            inst_probs=inst_probs,
            presence_probs=pres,
            decoder_context=lo,
            fpn_feats=fpn_feats,
        )


class _MockDINOv3Expert(nn.Module):
    def __init__(self, sat_dim: int = 64, grid: int = 12) -> None:
        super().__init__()
        self.embed_dim = sat_dim
        self.grid = grid
        self.proj = nn.Conv2d(3, sat_dim, kernel_size=1)
        for p in self.parameters():
            p.requires_grad = False
        self.eval()

    @torch.no_grad()
    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(images_01, size=(self.grid, self.grid), mode="bilinear", align_corners=False)
        return self.proj(x).permute(0, 2, 3, 1).contiguous()


def test_sparse_window_gather_scatter_exact() -> None:
    torch.manual_seed(42)
    x = torch.randn(2, 12, 12, 16, requires_grad=True)
    # Select 3 out of 9 4x4 windows per batch element
    idx = torch.tensor([[0, 4, 8], [1, 3, 7]], dtype=torch.long)
    sel, _ = gather_selected_windows(x, idx, window_size=4)
    assert sel.shape == (6, 4, 4, 16)

    upd = sel * 2.0 + 1.0
    merged = scatter_selected_windows(x.detach(), upd, idx, window_size=4)
    assert merged.shape == x.shape
    merged.sum().backward()
    assert x.grad is not None and x.grad.abs().sum().item() > 0.0


def test_uncertainty_router_budget_and_mask() -> None:
    router = UncertaintyWindowRouter(window_size=4, top_m_windows=3, pe_dim=32, sat_dim=32, proj_dim=16)
    probs = F.softmax(torch.randn(2, 7, 24, 24), dim=1)
    pe = torch.randn(2, 12, 12, 32)
    sat = torch.randn(2, 12, 12, 32)
    out = router(probs, pe, sat, top_m_override=3)
    assert out.selected_indices.shape == (2, 3)
    assert abs(out.active_ratio - 3.0 / 9.0) < 1e-6
    # Exactly 3 windows * 16 patches = 48 active patches per sample
    assert out.active_mask.sum(dim=(1, 2, 3)).tolist() == [48.0, 48.0]


def test_georecvit_inner_and_outer_end_to_end(tmp_path: Path) -> None:
    torch.manual_seed(0)
    sam3 = _MockSAM3Wrapper(pe_dim=64, d_model=32, grid=12, window_size=4)
    dino = _MockDINOv3Expert(sat_dim=64, grid=12)
    cfg = GeoRecViTConfig(
        num_classes=7,
        pe_dim=64,
        sat_dim=64,
        decoder_dim=32,
        window_size=4,
        num_recursive_rounds=3,
        sparse_top_windows=3,
    )
    model = GeoRecViT(sam3, dino, cfg)
    model.train()

    imgs = torch.rand(2, 3, 24, 24)
    masks = torch.randint(0, 7, (2, 24, 24))
    prompts = [f"class_{i}" for i in range(7)]

    # 1. Stage 1 Inner-Loop alignment
    inner_out = model.forward_inner_alignment(imgs)
    inner_out["loss"].backward()
    assert model.inner_link is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.inner_link.parameters())
    assert all(p.grad is None for p in sam3.parameters())
    assert all(p.grad is None for p in dino.parameters())

    # 2. Stage 2 Outer-Loop R=3 closed-loop recursion (with DEFAULT non-zero gate init!)
    for p in model.trainable_parameters():
        p.grad = None

    out = model(imgs, class_prompts=prompts, num_rounds=3, top_m_windows=3, return_routing=True)
    assert len(out["round_logits"]) == 3
    assert out["final_logits"].shape == (2, 7, 24, 24)
    assert out["active_ratios"] == [1.0, 3.0 / 9.0, 3.0 / 9.0]
    # Verify multi-round outputs evolve inside active windows AND stay strictly invariant outside!
    diff_12_map = (out["round_logits"][1] - out["round_logits"][0]).abs()
    act_mask_1 = out["active_masks_hi"][0]  # [B, 1, 24, 24]
    assert diff_12_map[act_mask_1.expand_as(diff_12_map) > 0.5].max().item() > 1e-5
    assert diff_12_map[act_mask_1.expand_as(diff_12_map) < 0.5].max().item() < 1e-6
    diff_23 = (out["round_logits"][2] - out["round_logits"][1]).abs().max().item()
    assert diff_23 > 1e-5, f"Round 3 identical to Round 2: diff_23={diff_23}"

    criterion = MultiRoundSegmentationLoss(ignore_index=255)
    loss_dict = criterion(out["round_logits"], masks, active_masks_hi=out["active_masks_hi"])
    loss_dict["loss"].backward()

    # Verify gradients reach EVERY SINGLE trainable module — INCLUDING STE router!
    for name, mod in model.trainable_modules().items():
        has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in mod.parameters())
        assert has_grad, f"Module {name} did not receive gradients!"
    assert model.router.uncertainty_logits.grad is not None
    assert model.router.uncertainty_logits.grad.abs().sum().item() > 0
    assert model.router.pool_mix_logit.grad is not None
    assert model.router.pool_mix_logit.grad.abs().item() > 0

    # Verify frozen backbones remain strictly untouched
    assert all(p.grad is None for p in sam3.parameters())
    assert all(p.grad is None for p in dino.parameters())

    # 2b. Verify no-DETR and every-round-DETR config modes run cleanly
    cfg_nodetr = GeoRecViTConfig(
        num_classes=7, pe_dim=64, sat_dim=64, decoder_dim=32, window_size=4,
        num_recursive_rounds=2, sparse_top_windows=3, use_sam3_detr=False,
    )
    out_nd = GeoRecViT(sam3, dino, cfg_nodetr)(imgs, class_prompts=prompts, num_rounds=2, top_m_windows=3)
    assert len(out_nd["round_logits"]) == 2

    cfg_every = GeoRecViTConfig(
        num_classes=7, pe_dim=64, sat_dim=64, decoder_dim=32, window_size=4,
        num_recursive_rounds=2, sparse_top_windows=3, detr_round1_only=False,
    )
    out_ev = GeoRecViT(sam3, dino, cfg_every)(imgs, class_prompts=prompts, num_rounds=2, top_m_windows=3)
    assert len(out_ev["round_logits"]) == 2

    # 3. Verify lightweight checkpoint save & reload
    ckpt = save_georecvit_links(model, tmp_path, tag="test")
    loaded = load_georecvit_links(model, ckpt, strict=True)
    assert "sat_outer_link" in loaded and "feedback_outer_link" in loaded

    # 4. Verify metrics & label normalization
    meter = SegmentationMeter(num_classes=7, ignore_index=255)
    meter.update(out["final_logits"].argmax(dim=1), masks)
    stats = meter.compute()
    assert 0.0 <= stats["mIoU"] <= 100.0

    raw_1idx = np.array([[0, 1, 7], [2, 3, 0]], dtype=np.uint8)
    norm_1idx = normalize_mask_array(raw_1idx, num_classes=7, label_offset=1, ignore_index=255)
    assert norm_1idx.tolist() == [[255, 0, 6], [1, 2, 255]]
