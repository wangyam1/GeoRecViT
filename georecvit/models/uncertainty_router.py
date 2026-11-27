"""Uncertainty-Gated Sparse Window Router for High-Resolution ViT Recursion.

Key structural insight from SAM 3 `vitdet.py`:
- SAM 3 PE-ViT at 1008x1008 with patch_size=14 produces a 72x72 token grid.
- Late layers `blocks[24..30]` use non-overlapping window attention with `window_size=24`.
- Since `72 / 24 = 3` exactly (zero padding!), the grid partitions natively into
  `3 x 3 = 9` independent 24x24 windows with exact pretrained 2D RoPE frequencies.

In remote sensing imagery, foreground structures (tiny vehicles, dense rooftops,
shadowed boundaries) occupy a small spatial fraction while background terrain is flat.
This router scores every patch & window using:
  1. Prediction Shannon Entropy `H(P^(r-1))` (aleatoric / boundary uncertainty)
  2. Cross-Expert Cosine Disagreement `1 - cos(F_pe, F_sat)` (domain-shift conflict
     between natural-image PE-ViT and satellite-pretrained DINOv3-SAT-493M)
and routes ONLY the Top-M most uncertain windows (e.g. M=3/9 = 33.3%) through the
frozen late ViT blocks in Round r >= 2, preserving exact 2D RoPE and slashing FLOPs.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RoutingOutput:
    """Outputs of uncertainty estimation and sparse window selection."""

    entropy_map: torch.Tensor         # [B, 1, H, W] normalized in [0, 1]
    disagreement_map: torch.Tensor    # [B, 1, H, W] normalized in [0, 1]
    uncertainty_map: torch.Tensor     # [B, 1, H, W] combined uncertainty score
    window_scores: torch.Tensor       # [B, num_win] mean+max score per 24x24 window
    soft_window_gates: torch.Tensor   # [B, num_win] differentiable soft window gates
    selected_indices: torch.Tensor    # [B, M] indices of selected windows in [0, num_win)
    active_mask: torch.Tensor         # [B, H, W, 1] STE binary mask (exact 0/1 fwd, smooth bwd)
    active_ratio: float               # M / num_win


class UncertaintyWindowRouter(nn.Module):
    """Computes hybrid uncertainty maps and selects Top-M native ViT windows."""

    def __init__(
        self,
        window_size: int = 24,
        top_m_windows: int = 3,
        pe_dim: int = 1024,
        sat_dim: int = 1024,
        proj_dim: int = 256,
    ) -> None:
        super().__init__()
        self.window_size = window_size
        self.top_m_windows = top_m_windows

        # Lightweight linear projections to measure cross-expert feature conflict
        # in a shared normalized subspace (without disturbing backbone weights).
        self.pe_proj = nn.Linear(pe_dim, proj_dim, bias=False)
        self.sat_proj = nn.Linear(sat_dim, proj_dim, bias=False)

        # Learnable mixing weights [entropy, disagreement] + [mean_pool, max_pool]
        self.uncertainty_logits = nn.Parameter(torch.tensor([0.5, 0.5]))
        self.pool_mix_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5
        self.gate_sharpness = 8.0

    @staticmethod
    def compute_normalized_entropy(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """Compute normalized Shannon entropy `[B, 1, H, W]` in `[0, 1]`."""
        num_classes = probs.shape[1]
        p = probs.clamp(min=eps, max=1.0 - eps)
        entropy = -(p * p.log()).sum(dim=1, keepdim=True)
        max_entropy = math.log(max(num_classes, 2))
        return (entropy / max_entropy).clamp(0.0, 1.0)

    def compute_expert_disagreement(
        self,
        pe_feats: torch.Tensor,
        sat_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        """Compute cosine conflict `[B, 1, H, W]` in `[0, 1]` between PE-ViT and DINOv3-SAT."""
        B, H, W, _ = pe_feats.shape
        if sat_feats is None:
            return torch.zeros((B, 1, H, W), device=pe_feats.device, dtype=pe_feats.dtype)

        p = F.normalize(self.pe_proj(pe_feats).float(), dim=-1, eps=1e-6)
        s = F.normalize(self.sat_proj(sat_feats).float(), dim=-1, eps=1e-6)
        cos_sim = (p * s).sum(dim=-1, keepdim=True)  # [B, H, W, 1] in [-1, 1]
        disagreement = 0.5 * (1.0 - cos_sim)         # [0, 1]
        return disagreement.permute(0, 3, 1, 2).contiguous()

    def forward(
        self,
        probs: torch.Tensor,
        pe_feats: torch.Tensor,
        sat_feats: torch.Tensor | None = None,
        top_m_override: int | None = None,
    ) -> RoutingOutput:
        """Score 24x24 windows on the 72x72 ViT grid and select Top-M active windows.

        Args:
            probs: `[B, C, H_p, W_p]` class probability map from Round (r-1).
            pe_feats: `[B, H, W, C_pe]` PE-ViT spatial features (typically 72x72).
            sat_feats: optional `[B, H, W, C_sat]` DINOv3-SAT features aligned to 72x72.
            top_m_override: optional override for M during test-time budget ablation.
        """
        B, H, W, _ = pe_feats.shape
        ws = self.window_size
        if H % ws != 0 or W % ws != 0:
            raise ValueError(
                f"ViT feature grid ({H}x{W}) must be divisible by window_size={ws}."
            )
        gh, gw = H // ws, W // ws
        num_windows = gh * gw

        if probs.shape[-2:] != (H, W):
            probs_grid = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
        else:
            probs_grid = probs

        entropy_map = self.compute_normalized_entropy(probs_grid)
        disagreement_map = self.compute_expert_disagreement(pe_feats, sat_feats)

        weights = F.softmax(self.uncertainty_logits, dim=0)
        uncertainty_map = weights[0] * entropy_map + weights[1] * disagreement_map

        # Partition [B, 1, H, W] -> [B, gh, ws, gw, ws] -> [B, num_windows, ws*ws]
        u_win = (
            uncertainty_map.squeeze(1)
            .view(B, gh, ws, gw, ws)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, num_windows, ws * ws)
        )
        mean_score = u_win.mean(dim=-1)
        max_score = u_win.amax(dim=-1)
        alpha = torch.sigmoid(self.pool_mix_logit)
        window_scores = alpha * mean_score + (1.0 - alpha) * max_score  # [B, num_windows]

        m = self.top_m_windows if top_m_override is None else int(top_m_override)
        m = max(1, min(m, num_windows))

        _, selected_indices = torch.topk(window_scores, k=m, dim=-1, largest=True, sorted=True)

        # Build Straight-Through Estimator (STE) binary window mask [B, H, W, 1]:
        # Forward pass is strictly sparse {0, 1}; backward pass delivers smooth gradients
        # into `uncertainty_logits`, `pool_mix_logit`, `pe_proj`, and `sat_proj`.
        hard_win_mask = torch.zeros((B, num_windows), device=pe_feats.device, dtype=torch.float32)
        hard_win_mask.scatter_(1, selected_indices, 1.0)
        sorted_scores, _ = torch.sort(window_scores.float(), dim=-1, descending=True)
        if m < num_windows:
            thresh = 0.5 * (sorted_scores[:, m - 1 : m] + sorted_scores[:, m : m + 1]).detach()
        else:
            thresh = (sorted_scores[:, -1:] - 0.5).detach()
        soft_win_gates = torch.sigmoid((window_scores.float() - thresh) * self.gate_sharpness)
        ste_win_mask = (hard_win_mask - soft_win_gates.detach()) + soft_win_gates

        active_mask = (
            ste_win_mask.view(B, gh, gw, 1, 1)
            .expand(B, gh, gw, ws, ws)
            .permute(0, 1, 3, 2, 4)
            .reshape(B, H, W, 1)
            .contiguous()
            .to(dtype=pe_feats.dtype)
        )

        return RoutingOutput(
            entropy_map=entropy_map,
            disagreement_map=disagreement_map,
            uncertainty_map=uncertainty_map,
            window_scores=window_scores,
            soft_window_gates=soft_win_gates,
            selected_indices=selected_indices,
            active_mask=active_mask,
            active_ratio=float(m) / float(num_windows),
        )


def gather_selected_windows(
    x: torch.Tensor,
    selected_indices: torch.Tensor,
    window_size: int = 24,
) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
    """Gather Top-M native windows `[B, H, W, C]` -> `[B * M, ws, ws, C]`.

    Because each selected window has shape `(ws, ws)` matching SAM 3's exact
    `window_size=24`, passing the gathered tensor directly into frozen SAM 3
    window blocks preserves exact pretrained 2D axial RoPE frequencies!
    """
    B, H, W, C = x.shape
    gh, gw = H // window_size, W // window_size
    num_win = gh * gw
    M = selected_indices.shape[1]

    # [B, num_win, ws, ws, C]
    windows = (
        x.view(B, gh, window_size, gw, window_size, C)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B, num_win, window_size, window_size, C)
    )
    gather_idx = (
        selected_indices[:, :, None, None, None]
        .expand(B, M, window_size, window_size, C)
    )
    selected = torch.gather(windows, dim=1, index=gather_idx)  # [B, M, ws, ws, C]
    return selected.reshape(B * M, window_size, window_size, C), (B, gh, gw)


def scatter_selected_windows(
    base_x: torch.Tensor,
    updated_windows: torch.Tensor,
    selected_indices: torch.Tensor,
    window_size: int = 24,
) -> torch.Tensor:
    """Differentiably scatter updated `[B*M, ws, ws, C]` windows back into `[B, H, W, C]`."""
    B, H, W, C = base_x.shape
    gh, gw = H // window_size, W // window_size
    num_win = gh * gw
    M = selected_indices.shape[1]

    base_windows = (
        base_x.view(B, gh, window_size, gw, window_size, C)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B, num_win, window_size, window_size, C)
    )
    upd = updated_windows.view(B, M, window_size, window_size, C).to(base_windows.dtype)
    scatter_idx = (
        selected_indices[:, :, None, None, None]
        .expand(B, M, window_size, window_size, C)
    )
    # Out-of-place scatter preserves clean autograd graph across recursion rounds
    merged = base_windows.scatter(dim=1, index=scatter_idx, src=upd)
    return (
        merged.view(B, gh, gw, window_size, window_size, C)
        .permute(0, 1, 3, 2, 4, 5)
        .reshape(B, H, W, C)
        .contiguous()
    )
