"""Pure Frozen DINOv3-SAT Baseline Segmenter (Baseline 1b `B` & Self-Recursion `B <-> B`).

Supports BOTH:
  - `num_rounds = 1` (`B` only): Strictly frozen DINOv3-SAT + fair multi-scale ASPP
    progressive 288x288 decoder (~3.74M trainable params, matching SAM 3 Alone 3.68M).
  - `num_rounds >= 2` (`B <-> B` homogeneous self-recursion): Re-enters frozen DINOv3-SAT
    late layers `layers[18:24]` with entropy-routed sparse feedback + CSPR residual refinement!

Directly answers BOTH reviewer questions:
  1. "Does SAM 3 + DINOv3-SAT genuinely achieve 1 + 1 > 2 over DINOv3-SAT alone?"
  2. "How does homogeneous B <-> B self-recursion compare against A <-> A and bidirectional A <-> B?"
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .georecvit_model import SparseResidualRefinementHead
from .recursive_links import FeedbackEncoderLink, SpatialInnerLink
from .uncertainty_router import UncertaintyWindowRouter


class DINOv3OnlySegmenter(nn.Module):
    """Frozen DINOv3-SAT backbone + multi-scale decoder + optional `B <-> B` self-recursion."""

    def __init__(
        self,
        dinov3_expert: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_size: int = 288,
        num_rounds: int = 1,
        window_size: int = 24,
        top_m_windows: int = 3,
        anchor_perturb_prob: float = 0.0,
    ) -> None:
        super().__init__()
        self.dinov3_expert = dinov3_expert
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.out_size = out_size
        self.num_rounds = max(1, int(num_rounds))
        self.top_m_windows = int(top_m_windows)
        self.anchor_perturb_prob = float(anchor_perturb_prob)
        sat_dim = int(getattr(dinov3_expert, "embed_dim", 1024))

        self.in_proj = nn.Sequential(
            nn.Conv2d(sat_dim, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )

        # Multi-scale ASPP context (1x1 + dilations 1, 2, 4 + global image context)
        b_dim = hidden_dim // 2
        self.aspp_1 = nn.Sequential(
            nn.Conv2d(hidden_dim, b_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, b_dim),
            nn.GELU(),
        )
        self.aspp_d1 = nn.Sequential(
            nn.Conv2d(hidden_dim, b_dim, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(8, b_dim),
            nn.GELU(),
        )
        self.aspp_d2 = nn.Sequential(
            nn.Conv2d(hidden_dim, b_dim, kernel_size=3, padding=2, dilation=2, bias=False),
            nn.GroupNorm(8, b_dim),
            nn.GELU(),
        )
        self.aspp_d4 = nn.Sequential(
            nn.Conv2d(hidden_dim, b_dim, kernel_size=3, padding=4, dilation=4, bias=False),
            nn.GroupNorm(8, b_dim),
            nn.GELU(),
        )
        self.aspp_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden_dim, b_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, b_dim),
            nn.GELU(),
        )
        self.aspp_fuse = nn.Sequential(
            nn.Conv2d(b_dim * 5, hidden_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(0.1),
        )

        # Progressive spatial refinement: 72x72 -> 144x144 -> 288x288 (exact SAM 3 FPN resolution)
        self.up1 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.cls_head = nn.Conv2d(hidden_dim, num_classes, kernel_size=1)

        # Optional B <-> B Self-Recursion modules (only instantiated when num_rounds > 1)
        if self.num_rounds > 1:
            self.router = UncertaintyWindowRouter(
                window_size=window_size,
                top_m_windows=self.top_m_windows,
                pe_dim=sat_dim,
                sat_dim=sat_dim,
            )
            self.sat_feedback_link = FeedbackEncoderLink(
                num_classes=num_classes,
                context_dim=hidden_dim,
                out_dim=sat_dim,
            )
            self.sat_inner_link = SpatialInnerLink(
                hidden_size=sat_dim,
                bottleneck_ratio=0.5,
                init_scale=0.1,
            )
            self.sparse_refiner = SparseResidualRefinementHead(
                num_classes=num_classes,
                fpn_dim=hidden_dim,
                sat_dim=sat_dim,
                proto_dim=hidden_dim,
            )
            self.register_buffer("zero_prototypes", torch.zeros(num_classes, hidden_dim))
        else:
            self.router = None
            self.sat_feedback_link = None
            self.sat_inner_link = None
            self.sparse_refiner = None

    def freeze_round1_anchor(self) -> None:
        """Freeze Round-1 decoder for decoupled Stage 2b `B <-> B` self-recursion."""
        for mod in (
            self.in_proj,
            self.aspp_1,
            self.aspp_d1,
            self.aspp_d2,
            self.aspp_d4,
            self.aspp_pool,
            self.aspp_fuse,
            self.up1,
            self.up2,
            self.cls_head,
        ):
            mod.eval()
            for p in mod.parameters():
                p.requires_grad = False
        self._round1_anchor_frozen = True

    def train(self, mode: bool = True) -> "DINOv3OnlySegmenter":
        super().train(mode)
        self.dinov3_expert.eval()
        if getattr(self, "_round1_anchor_frozen", False):
            for mod in (
                self.in_proj,
                self.aspp_1,
                self.aspp_d1,
                self.aspp_d2,
                self.aspp_d4,
                self.aspp_pool,
                self.aspp_fuse,
                self.up1,
                self.up2,
                self.cls_head,
            ):
                mod.eval()
        return self

    def trainable_parameters(self) -> List[nn.Parameter]:
        return [p for p in self.parameters() if p.requires_grad]

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }

    def _decode_sat(self, f_sat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = f_sat.permute(0, 3, 1, 2).contiguous().float()
        h0 = self.in_proj(x)
        H, W = h0.shape[-2:]
        p = F.interpolate(self.aspp_pool(h0), size=(H, W), mode="bilinear", align_corners=False)
        ctx = self.aspp_fuse(
            torch.cat(
                [self.aspp_1(h0), self.aspp_d1(h0), self.aspp_d2(h0), self.aspp_d4(h0), p],
                dim=1,
            )
        )
        mid_sz = self.out_size // 2
        h = F.interpolate(ctx, size=(mid_sz, mid_sz), mode="bilinear", align_corners=False)
        h = h + self.up1(h)
        h = F.interpolate(h, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        h = h + self.up2(h)
        logits = self.cls_head(h)
        return logits, h, ctx

    def forward(
        self,
        images_01: torch.Tensor,
        num_rounds: Optional[int] = None,
        top_m_windows: Optional[int] = None,
    ) -> Dict[str, object]:
        R = int(num_rounds if num_rounds is not None else self.num_rounds)
        R = max(1, R)
        r1_frozen = bool(getattr(self, "_round1_anchor_frozen", False))
        device = images_01.device
        perturb_r1 = (
            self.training
            and r1_frozen
            and R > 1
            and self.anchor_perturb_prob > 0.0
            and (torch.rand((), device=device).item() < self.anchor_perturb_prob)
        )

        use_split = (R > 1) and hasattr(self.dinov3_expert, "forward_early")
        if use_split:
            sat_early = self.dinov3_expert.forward_early(images_01)
            with torch.no_grad():
                f_sat = self.dinov3_expert.forward_late(sat_early)
        else:
            sat_early = None
            f_sat = self.dinov3_expert(images_01)

        if r1_frozen:
            with torch.no_grad():
                f_sat_r1 = (
                    f_sat + 0.15 * torch.randn_like(f_sat)
                    if perturb_r1
                    else f_sat
                )
                logits_1, feat_hi_1, ctx_1 = self._decode_sat(f_sat_r1)
                if perturb_r1:
                    logits_1 = logits_1 + 0.25 * torch.randn_like(logits_1)
        else:
            logits_1, feat_hi_1, ctx_1 = self._decode_sat(f_sat)

        round_logits: List[torch.Tensor] = [logits_1]
        active_masks_hi: List[torch.Tensor] = []
        active_ratios: List[float] = [1.0]

        if R == 1 or self.router is None or self.sat_feedback_link is None or self.sparse_refiner is None:
            return {
                "round_logits": round_logits,
                "final_logits": logits_1,
                "active_ratios": active_ratios,
                "active_masks_hi": active_masks_hi,
            }

        prev_logits = logits_1
        prev_probs = F.softmax(logits_1, dim=1)
        prev_feat_hi = feat_hi_1
        prev_ctx = ctx_1
        cur_sat = f_sat
        cur_sat_delta: Optional[torch.Tensor] = None

        for r in range(1, R):
            routing = self.router(
                probs=prev_probs,
                pe_feats=cur_sat,
                sat_feats=None,
                top_m_override=top_m_windows,
            )
            active_ratios.append(routing.active_ratio)

            fb_tokens = self.sat_feedback_link(
                probs=prev_probs,
                entropy=routing.entropy_map,
                disagreement=routing.disagreement_map,
                decoder_context=prev_ctx,
                round_idx=r,
            )  # [B, 72, 72, sat_dim]
            fb_masked = fb_tokens * routing.active_mask

            if use_split and sat_early is not None:
                # Align feedback into DINOv3-SAT mid manifold and re-run frozen layers[18:24]
                fb_lo = F.interpolate(
                    fb_masked.permute(0, 3, 1, 2).float(),
                    size=(sat_early.h_p, sat_early.w_p),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                mask_lo = F.interpolate(
                    routing.active_mask.permute(0, 3, 1, 2).float(),
                    size=(sat_early.h_p, sat_early.w_p),
                    mode="bilinear",
                    align_corners=False,
                ).permute(0, 2, 3, 1)
                aligned_in = self.sat_inner_link(sat_early.spatial_mid.float() + fb_lo) - sat_early.spatial_mid.float()
                step_delta = aligned_in * mask_lo
                cur_sat_delta = step_delta if cur_sat_delta is None else (cur_sat_delta + step_delta)
                cur_sat = self.dinov3_expert.forward_late(sat_early, spatial_delta=cur_sat_delta)
            else:
                cur_sat = cur_sat + fb_masked.to(cur_sat.dtype)

            # Decode updated high-res representation from updated DINOv3-SAT features
            _, cur_feat_hi, cur_ctx = self._decode_sat(cur_sat)
            logits_r = self.sparse_refiner(
                prev_logits=prev_logits,
                fpn_high=cur_feat_hi,
                prev_fpn_high=prev_feat_hi,
                sat_up=cur_feat_hi,
                prev_probs=prev_probs,
                entropy_map=routing.entropy_map,
                disagreement_map=routing.disagreement_map,
                active_mask_grid=routing.active_mask,
                class_prototypes=self.zero_prototypes,
                round_idx=r,
                f_sat=cur_sat,
            )
            H_hi, W_hi = cur_feat_hi.shape[-2:]
            active_masks_hi.append(
                F.interpolate(
                    routing.active_mask.permute(0, 3, 1, 2).detach().float(),
                    size=(H_hi, W_hi),
                    mode="nearest",
                )
            )
            round_logits.append(logits_r)
            prev_logits = logits_r
            prev_probs = F.softmax(logits_r, dim=1)
            prev_feat_hi = cur_feat_hi
            prev_ctx = cur_ctx

        return {
            "round_logits": round_logits,
            "final_logits": round_logits[-1],
            "active_ratios": active_ratios,
            "active_masks_hi": active_masks_hi,
        }
