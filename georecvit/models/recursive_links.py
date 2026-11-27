"""Spatial RecursiveLink modules adapted from RecursiveMAS (NeurIPS 2026).

Design correspondence with RecursiveMAS (`train/model.py`):
- `SpatialInnerLink`: mirrors `Adapter(adapter_type="ln_res_adapter")`
  (`Pre-LN -> Linear -> GELU -> Linear + Residual -> Post-LN`), calibrated in Stage 1
  (Inner-Loop Warm-start) so injected heterogeneous / feedback tokens remain strictly
  in-distribution with frozen SAM 3 PE-ViT intermediate feature manifolds.
- `SpatialOuterLink`: mirrors `CrossModelAdapter(adapter_type="outer_ln_res_adapter")`
  (`LN_src -> Linear(in, 2*out) -> GELU -> Linear(2*out, out) + ResidualProj -> LN_tgt`)
  with a learnable zero-initialized residual gate so Round 1 starts smoothly from the
  pretrained frozen SAM 3 manifold without corrupting zero-shot concept semantics.
- `FeedbackEncoderLink`: compresses Round-(r-1) Decoder/Segmentation Critic state
  (multi-class probabilities, pixel Shannon entropy, PE/DINO disagreement, and pooled
  decoder spatial context) back into the 1024-d PE-ViT latent space for closed-loop
  top-down recursion.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialInnerLink(nn.Module):
    """In-distribution manifold aligner before frozen ViT late blocks.

    Directly adapted from RecursiveMAS `ln_res_adapter` with an optional
    learnable residual scale for stable warm-start into frozen ViT layers.
    """

    def __init__(
        self,
        hidden_size: int = 1024,
        bottleneck_ratio: float = 0.5,
        init_scale: float = 0.1,
    ) -> None:
        super().__init__()
        mid_dim = max(64, int(hidden_size * bottleneck_ratio))
        self.hidden_size = hidden_size
        self.pre_ln = nn.LayerNorm(hidden_size)
        self.proj1 = nn.Linear(hidden_size, mid_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(mid_dim, hidden_size)
        self.post_ln = nn.LayerNorm(hidden_size)
        self.res_scale = nn.Parameter(torch.full((hidden_size,), init_scale))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Align tokens `[..., C]` into the frozen ViT block manifold."""
        h = self.pre_ln(x)
        delta = self.proj2(self.act(self.proj1(h)))
        out = x + self.res_scale * delta
        return self.post_ln(out)


class SpatialOuterLink(nn.Module):
    """Cross-expert latent bridge (e.g. DINOv3-SAT 1024-d -> SAM 3 PE-ViT 1024-d).

    Directly adapted from RecursiveMAS `CrossModelAdapter` (`outer_ln_res_adapter`),
    augmented with an optional depthwise 3x3 local spatial context refinement and a
    zero-initialized residual gate for smooth warm-start on frozen backbones.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: float = 1.0,
        use_spatial_dwconv: bool = True,
        zero_init_gate: bool = False,
        init_gate_value: float = 0.1,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        hidden_dim = max(128, int(out_dim * hidden_mult))

        self.ln_source = nn.LayerNorm(in_dim)
        self.proj1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.proj2 = nn.Linear(hidden_dim, out_dim)
        self.residual_proj = nn.Linear(in_dim, out_dim)
        self.ln_target = nn.LayerNorm(out_dim)

        self.use_spatial_dwconv = use_spatial_dwconv
        if use_spatial_dwconv:
            self.dwconv = nn.Conv2d(
                out_dim, out_dim, kernel_size=3, padding=1, groups=out_dim
            )
        else:
            self.dwconv = None

        init_val = 0.0 if zero_init_gate else float(init_gate_value)
        self.gate = nn.Parameter(torch.full((out_dim,), init_val))

    def forward(self, x: torch.Tensor, apply_gate: bool = True) -> torch.Tensor:
        """Map source features `[B, H, W, C_in]` or `[..., C_in]` to target space."""
        h = self.ln_source(x)
        out = self.proj2(self.act(self.proj1(h))) + self.residual_proj(x)
        if self.dwconv is not None and out.ndim == 4:
            # [B, H, W, C] -> [B, C, H, W] -> depthwise local spatial context
            spatial = self.dwconv(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
            out = out + spatial
        out = self.ln_target(out)
        if apply_gate:
            out = self.gate * out
        return out


class FeedbackEncoderLink(nn.Module):
    """Closed-loop Feedback OuterLink: SAM 3 Decoder/Seg Critic -> PE-ViT 1024-d.

    Encodes Round-(r-1) segmentation hypotheses & uncertainty signals into top-down
    latent feedback tokens injected into the sparse high-uncertainty ViT windows:
      - Multi-class probability distribution `P^(r-1)`: `[B, C, 72, 72]`
      - Uncertainty cues `[entropy, disagreement]`: `[B, 2, 72, 72]`
      - Optional SAM 3 encoder/decoder visual context `E^(r-1)`: `[B, D_ctx, 72, 72]`
      - Round step embedding `e_r` (tells the bridge which recursion depth it is at)
    """

    def __init__(
        self,
        num_classes: int,
        context_dim: int = 256,
        out_dim: int = 1024,
        max_rounds: int = 6,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.context_dim = context_dim
        self.out_dim = out_dim

        in_channels = num_classes + 2 + context_dim
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(in_channels, context_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, context_dim),
            nn.GELU(),
            nn.Conv2d(context_dim, context_dim, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(8, context_dim),
            nn.GELU(),
        )
        self.outer_bridge = SpatialOuterLink(
            in_dim=context_dim,
            out_dim=out_dim,
            hidden_mult=1.0,
            use_spatial_dwconv=False,
            zero_init_gate=False,
            init_gate_value=0.1,
        )
        self.round_embed = nn.Embedding(max_rounds, out_dim)
        nn.init.normal_(self.round_embed.weight, std=0.02)

    def forward(
        self,
        probs: torch.Tensor,
        entropy: torch.Tensor,
        disagreement: torch.Tensor,
        decoder_context: torch.Tensor,
        round_idx: int,
    ) -> torch.Tensor:
        """Return top-down latent feedback `[B, H, W, out_dim]` at ViT grid size (72x72)."""
        target_hw = decoder_context.shape[-2:]
        if probs.shape[-2:] != target_hw:
            probs = F.interpolate(probs, size=target_hw, mode="bilinear", align_corners=False)
        if entropy.shape[-2:] != target_hw:
            entropy = F.interpolate(entropy, size=target_hw, mode="bilinear", align_corners=False)
        if disagreement.shape[-2:] != target_hw:
            disagreement = F.interpolate(
                disagreement, size=target_hw, mode="bilinear", align_corners=False
            )

        stacked = torch.cat([probs, entropy, disagreement, decoder_context], dim=1)
        feat = self.spatial_stem(stacked).permute(0, 2, 3, 1)  # [B, H, W, context_dim]
        fb = self.outer_bridge(feat, apply_gate=True)

        r_idx = min(max(round_idx, 0), self.round_embed.num_embeddings - 1)
        r_vec = self.round_embed.weight[r_idx].view(1, 1, 1, -1)
        # Scale round embedding by the same zero-init gate magnitude so step-0 stays clean
        return fb + self.outer_bridge.gate * r_vec
