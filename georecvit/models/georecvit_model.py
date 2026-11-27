"""GeoRecViT: Heterogeneous Recursive ViT with SAM 3 + DINOv3-SAT for Remote Sensing.

Combines four core ideas from RecursiveMAS (NeurIPS 2026) adapted natively to 2D ViTs:
1. Heterogeneous Latent Collaboration:
   - Expert 1: Frozen SAM 3 PE-ViT (natural-image open-vocabulary concept backbone)
   - Expert 2: Frozen DINOv3-ViT-L/16-SAT493M (493M satellite structural geometry expert)
   - Expert 3: Frozen SAM 3 DETR Encoder + Decoder + UniversalSegmentationHead
2. Lightweight Frozen-Backbone `RecursiveLink` Modules (~12M trainable params total):
   - `SpatialOuterLink` (DINOv3-SAT -> PE-ViT)
   - `FeedbackEncoderLink` (SAM 3 Critic -> PE-ViT)
   - `SpatialInnerLink` (in-distribution manifold aligner before `blocks[24]`)
3. Uncertainty-Gated Sparse Late-ViT Closed-Loop Recursion:
   - Round 1: Full `blocks[24:32]` pass -> initial logits + hybrid uncertainty map
   - Round r >= 2: Routes ONLY Top-M uncertain native 24x24 windows through frozen
     `blocks[24:31]` + global `block[31]` -> refined concept predictions
4. Inner-Outer Loop Two-Stage Training:
   - Stage 1 (`forward_inner_alignment`): aligns DINOv3-SAT + feedback projections into
     PE-ViT block-24 manifold (Cosine + MSE loss).
   - Stage 2 (`forward`): unrolls R recursive rounds with shared gradient credit
     assignment across rounds (`L_final + lambda * mean(L_prev)`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from .recursive_links import FeedbackEncoderLink, SpatialInnerLink, SpatialOuterLink
from .uncertainty_router import RoutingOutput, UncertaintyWindowRouter


@dataclass
class GeoRecViTConfig:
    num_classes: int = 7
    pe_dim: int = 1024
    sat_dim: int = 1024
    decoder_dim: int = 256
    window_size: int = 24
    num_recursive_rounds: int = 3
    sparse_top_windows: int = 3
    use_sam3_detr: bool = True
    detr_round1_only: bool = True
    use_dinov3: bool = True
    use_inner_link: bool = True
    use_feedback_loop: bool = True


class ConceptCalibrationReadout(nn.Module):
    """Open-vocabulary dual-head calibration combining SAM 3 priors + refined FPN + SAT cues.

    Uses bounded calibration over SAM 3's pretrained `sem_logits` and `inst_probs` (avoiding
    singular `logit(0)` penalties on stuff classes like bareland/rangeland) combined with
    a multi-scale refined visual-concept head anchored to SAM 3's `VETextEncoder` prototypes.
    """

    def __init__(
        self,
        num_classes: int,
        fpn_dim: int = 256,
        sat_dim: int = 1024,
        proto_dim: int = 256,
    ) -> None:
        super().__init__()
        self.fpn_norm = nn.GroupNorm(8, fpn_dim)
        self.sat_to_fpn = nn.Sequential(
            nn.Conv2d(sat_dim, fpn_dim, kernel_size=1, bias=False),
            nn.GroupNorm(8, fpn_dim),
            nn.GELU(),
        )
        # Fuse refined PE-ViT FPN + satellite structural features
        self.refine_conv = nn.Sequential(
            nn.Conv2d(fpn_dim * 2, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, fpn_dim),
            nn.GELU(),
            nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, fpn_dim),
            nn.GELU(),
        )
        # Text-anchored open-vocab projection + lightweight per-class residual adapter
        self.proto_proj = nn.Conv2d(fpn_dim, proto_dim, kernel_size=1)
        self.class_residual = nn.Conv2d(fpn_dim, num_classes, kernel_size=1)
        nn.init.zeros_(self.class_residual.weight)
        nn.init.zeros_(self.class_residual.bias)

        # Bounded, well-scaled fusion weights so Step-0 preserves Baseline 0 zero-shot quality
        self.sem_scale = nn.Parameter(torch.tensor(0.35))
        self.inst_scale = nn.Parameter(torch.tensor(1.0))
        self.proto_scale = nn.Parameter(torch.tensor(1.0))
        self.logit_temp = nn.Parameter(torch.tensor(6.0))
        self.class_bias = nn.Parameter(torch.zeros(1, num_classes, 1, 1))

    def forward(
        self,
        sam3_sem_logits: torch.Tensor,
        sam3_inst_probs: torch.Tensor,
        fpn_high: torch.Tensor,
        sat_feats_grid: Optional[torch.Tensor],
        class_prototypes: torch.Tensor,
    ) -> torch.Tensor:
        """Fuse SAM 3 dual-head priors with refined ViT + SAT visual logits -> `[B, C, 288, 288]`."""
        H, W = fpn_high.shape[-2:]
        fpn_normed = self.fpn_norm(fpn_high)
        if sat_feats_grid is not None:
            sat_map = sat_feats_grid.permute(0, 3, 1, 2).contiguous()
            sat_proj = self.sat_to_fpn(sat_map)
            sat_up = F.interpolate(sat_proj, size=(H, W), mode="bilinear", align_corners=False)
        else:
            sat_up = torch.zeros_like(fpn_normed)

        refined = self.refine_conv(torch.cat([fpn_normed, sat_up], dim=1))
        visual_embed = F.normalize(self.proto_proj(refined).float(), dim=1, eps=1e-6)
        proto = F.normalize(class_prototypes.float().to(device=refined.device), dim=-1, eps=1e-6)
        proto_logits = torch.einsum("bchw,kc->bkhw", visual_embed, proto) * self.logit_temp.float()
        res_logits = self.class_residual(refined).float()

        # Clamp SAM 3 raw semantic logits gently to avoid extreme softmax saturation on stuff classes
        sem_bounded = sam3_sem_logits.float().clamp(-10.0, 10.0)
        inst_centered = 2.0 * sam3_inst_probs.float() - 1.0  # bounded in [-1, 1]

        return (
            self.sem_scale.float() * sem_bounded
            + self.inst_scale.float() * inst_centered
            + self.proto_scale.float() * proto_logits
            + res_logits
            + self.class_bias.float()
        )


class GeoRecViT(nn.Module):
    """Full GeoRecViT closed-loop architecture with frozen SAM 3 + frozen DINOv3-SAT."""

    def __init__(
        self,
        sam3_wrapper: nn.Module,
        dinov3_expert: Optional[nn.Module],
        config: GeoRecViTConfig,
    ) -> None:
        super().__init__()
        self.sam3_wrapper = sam3_wrapper
        self.dinov3_expert = dinov3_expert if config.use_dinov3 else None
        self.cfg = config

        # 1. Cross-expert OuterLink: DINOv3-SAT -> SAM 3 PE-ViT (1024 -> 1024)
        self.sat_outer_link = (
            SpatialOuterLink(
                in_dim=config.sat_dim,
                out_dim=config.pe_dim,
                hidden_mult=1.0,
                use_spatial_dwconv=True,
                zero_init_gate=False,
                init_gate_value=0.1,
            )
            if config.use_dinov3
            else None
        )

        # 2. Feedback OuterLink: SAM 3 Decoder/Seg Critic -> PE-ViT (256+C+2 -> 1024)
        self.feedback_outer_link = (
            FeedbackEncoderLink(
                num_classes=config.num_classes,
                context_dim=config.decoder_dim,
                out_dim=config.pe_dim,
            )
            if config.use_feedback_loop
            else None
        )

        # 3. Spatial InnerLink: in-distribution manifold aligner before PE-ViT blocks[24]
        self.inner_link = (
            SpatialInnerLink(hidden_size=config.pe_dim, bottleneck_ratio=0.5, init_scale=0.1)
            if config.use_inner_link
            else None
        )

        # 4. Uncertainty-Gated Sparse Window Router
        self.router = UncertaintyWindowRouter(
            window_size=config.window_size,
            top_m_windows=config.sparse_top_windows,
            pe_dim=config.pe_dim,
            sat_dim=config.sat_dim,
        )

        # 5. Text-anchored Open-Vocabulary Concept Calibration Readout
        self.readout = ConceptCalibrationReadout(
            num_classes=config.num_classes,
            fpn_dim=config.decoder_dim,
            sat_dim=config.sat_dim,
            proto_dim=config.decoder_dim,
        )

    def trainable_modules(self) -> Dict[str, nn.Module]:
        """Return dict of ONLY the lightweight trainable modules (~10-15M params)."""
        mods: Dict[str, nn.Module] = {
            "router": self.router,
            "readout": self.readout,
        }
        if self.sat_outer_link is not None:
            mods["sat_outer_link"] = self.sat_outer_link
        if self.feedback_outer_link is not None:
            mods["feedback_outer_link"] = self.feedback_outer_link
        if self.inner_link is not None:
            mods["inner_link"] = self.inner_link
        return mods

    def trainable_parameters(self) -> List[nn.Parameter]:
        params: List[nn.Parameter] = []
        for m in self.trainable_modules().values():
            params.extend(p for p in m.parameters() if p.requires_grad)
        return params

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.trainable_parameters())
        return {
            "total_params": total,
            "trainable_params": trainable,
            "frozen_params": total - trainable,
        }

    def train(self, mode: bool = True) -> "GeoRecViT":
        super().train(mode)
        # Enforce frozen eval mode on both foundation experts
        self.sam3_wrapper.eval()
        if self.dinov3_expert is not None:
            self.dinov3_expert.eval()
        return self

    def forward_inner_alignment(
        self,
        images_01: torch.Tensor,
        cos_weight: float = 1.0,
        mse_weight: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Stage 1 (Inner-Loop Warm-Start, cf. RecursiveMAS `train_inner.py`).

        Trains `sat_outer_link` (ungated projection) + `inner_link` so that combining
        frozen `F_pe^(24)` with `DINOv3-SAT` features maps smoothly back into the
        pretrained `F_pe^(24)` statistical manifold without causing OOD drift in frozen
        late ViT blocks `blocks[24:32]`.
        """
        f_pe_mid = self.sam3_wrapper.forward_early_vit(images_01)  # [B, 72, 72, 1024]
        if self.dinov3_expert is not None and self.sat_outer_link is not None:
            f_sat = self.dinov3_expert(images_01).to(dtype=f_pe_mid.dtype)
            # Align ungated SAT projection directly towards PE-ViT mid manifold
            sat_mapped = self.sat_outer_link(f_sat, apply_gate=False)
            sat_cos = (1.0 - F.cosine_similarity(sat_mapped.float(), f_pe_mid.float(), dim=-1)).mean()
            sat_mse = F.mse_loss(sat_mapped.float(), f_pe_mid.float())
            mixed = f_pe_mid + 0.5 * sat_mapped
        else:
            sat_cos = torch.zeros((), device=f_pe_mid.device)
            sat_mse = torch.zeros((), device=f_pe_mid.device)
            mixed = f_pe_mid

        if self.inner_link is not None:
            aligned = self.inner_link(mixed)
            inner_cos = (1.0 - F.cosine_similarity(aligned.float(), f_pe_mid.float(), dim=-1)).mean()
            inner_mse = F.mse_loss(aligned.float(), f_pe_mid.float())
        else:
            inner_cos = torch.zeros((), device=f_pe_mid.device)
            inner_mse = torch.zeros((), device=f_pe_mid.device)

        total_loss = (
            cos_weight * (sat_cos + inner_cos) + mse_weight * (sat_mse + inner_mse)
        )
        return {
            "loss": total_loss,
            "sat_cos_loss": sat_cos.detach(),
            "sat_mse_loss": sat_mse.detach(),
            "inner_cos_loss": inner_cos.detach(),
            "inner_mse_loss": inner_mse.detach(),
        }

    def forward(
        self,
        images_01: torch.Tensor,
        class_prompts: Sequence[str],
        num_rounds: Optional[int] = None,
        top_m_windows: Optional[int] = None,
        return_routing: bool = False,
    ) -> Dict[str, object]:
        """Stage 2 / Inference: R-round Closed-Loop Heterogeneous Sparse ViT Recursion.

        Returns:
            dict with:
              - `round_logits`: List of `[B, C, 288, 288]` logits for Round 1..R
              - `final_logits`: `[B, C, 288, 288]` logits from final Round R
              - `sam3_raw_logits`: `[B, C, 288, 288]` Round-1 raw SAM 3 semantic logits
              - `active_ratios`: List of active window ratios per recursive round
              - `routing_history`: optional List[RoutingOutput]
        """
        R = int(num_rounds if num_rounds is not None else self.cfg.num_recursive_rounds)
        R = max(1, R)
        device = images_01.device

        # Step 0 (Cached ONCE per image): frozen PE-ViT early trunk + frozen DINOv3-SAT + text prompts
        f_pe_mid = self.sam3_wrapper.forward_early_vit(images_01)  # [B, 72, 72, 1024]
        f_sat = (
            self.dinov3_expert(images_01).to(dtype=f_pe_mid.dtype)
            if self.dinov3_expert is not None
            else None
        )
        text_out = self.sam3_wrapper.encode_class_prompts(class_prompts, device=device)
        class_prototypes = text_out["class_prototypes"]

        # Inject heterogeneous satellite structural features via OuterLink
        sat_delta = (
            self.sat_outer_link(f_sat, apply_gate=True)
            if (self.sat_outer_link is not None and f_sat is not None)
            else None
        )
        base_mid = f_pe_mid if sat_delta is None else (f_pe_mid + sat_delta)
        cur_mid = base_mid

        round_logits: List[torch.Tensor] = []
        active_ratios: List[float] = []
        routing_history: List[RoutingOutput] = []

        prev_pre_global: Optional[torch.Tensor] = None
        prev_decode = None
        cached_detr = None
        prev_probs: Optional[torch.Tensor] = None
        sam3_raw_logits: Optional[torch.Tensor] = None

        for r in range(R):
            if r == 0:
                x_in = self.inner_link(cur_mid) if self.inner_link is not None else cur_mid
                late_out = self.sam3_wrapper.forward_late_vit(x_in)
                active_ratios.append(1.0)
            else:
                assert prev_probs is not None and prev_decode is not None
                routing = self.router(
                    probs=prev_probs,
                    pe_feats=prev_pre_global,
                    sat_feats=f_sat,
                    top_m_override=top_m_windows,
                )
                routing_history.append(routing)
                active_ratios.append(routing.active_ratio)

                if self.feedback_outer_link is not None:
                    fb_tokens = self.feedback_outer_link(
                        probs=prev_probs,
                        entropy=routing.entropy_map,
                        disagreement=routing.disagreement_map,
                        decoder_context=prev_decode.decoder_context,
                        round_idx=r,
                    )
                    # Accumulate top-down latent feedback strictly inside active uncertain windows
                    cur_mid = cur_mid + fb_tokens * routing.active_mask

                x_in = self.inner_link(cur_mid) if self.inner_link is not None else cur_mid
                late_out = self.sam3_wrapper.forward_late_vit(
                    x_mid=x_in,
                    selected_indices=routing.selected_indices,
                    prev_pre_global=prev_pre_global,
                )

            prev_pre_global = late_out.pre_global
            fpn_feats, fpn_pos = self.sam3_wrapper.forward_neck(late_out.vit_out)

            # Fast asymmetric closed-loop: run full SAM 3 DETR on Round 1 to obtain open-vocab
            # semantic/instance/presence priors; in Rounds r >= 2 reuse cached DETR priors while
            # updating spatial decoder_context & high-res FPN via sparse late-ViT recursion!
            if not self.cfg.use_sam3_detr:
                decode_out = self.sam3_wrapper.forward_concept_decode(
                    fpn_feats=fpn_feats,
                    fpn_pos=fpn_pos,
                    text_out=text_out,
                    run_detr=False,
                )
            elif (r == 0) or (not self.cfg.detr_round1_only):
                if self.cfg.detr_round1_only and r == 0:
                    with torch.no_grad():
                        decode_out = self.sam3_wrapper.forward_concept_decode(
                            fpn_feats=[f.detach() for f in fpn_feats],
                            fpn_pos=fpn_pos,
                            text_out=text_out,
                            run_detr=True,
                        )
                    # Re-attach differentiable FPN from late ViT so Round 1 grads flow through late ViT!
                    decode_out.fpn_feats = fpn_feats
                else:
                    decode_out = self.sam3_wrapper.forward_concept_decode(
                        fpn_feats=fpn_feats,
                        fpn_pos=fpn_pos,
                        text_out=text_out,
                        run_detr=True,
                    )
                cached_detr = decode_out
            else:
                assert cached_detr is not None
                # Combine Round-1 SAM 3 DETR concept priors with Round-r refined ViT FPN!
                decode_out = type(cached_detr)(
                    sem_logits=cached_detr.sem_logits,
                    inst_probs=cached_detr.inst_probs,
                    presence_probs=cached_detr.presence_probs,
                    decoder_context=0.5 * cached_detr.decoder_context + 0.5 * fpn_feats[-1],
                    fpn_feats=fpn_feats,
                )

            if r == 0:
                sam3_raw_logits = decode_out.sem_logits.detach()

            logits_r = self.readout(
                sam3_sem_logits=decode_out.sem_logits,
                sam3_inst_probs=decode_out.inst_probs,
                fpn_high=decode_out.fpn_feats[0],
                sat_feats_grid=f_sat,
                class_prototypes=class_prototypes,
            )
            round_logits.append(logits_r)
            prev_decode = decode_out
            prev_probs = F.softmax(logits_r, dim=1)

        out: Dict[str, object] = {
            "round_logits": round_logits,
            "final_logits": round_logits[-1],
            "sam3_raw_logits": sam3_raw_logits,
            "active_ratios": active_ratios,
        }
        if return_routing:
            out["routing_history"] = routing_history
        return out
