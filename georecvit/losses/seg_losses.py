"""Multi-Round Outer-Loop Segmentation Loss for GeoRecViT.

Mirrors RecursiveMAS `train/outer/sequential.py` cross-round credit assignment:
    L_total = L_seg(Y^(R)) + lambda * mean_{r < R} L_seg(Y^(r))
where each round's segmentation loss combines:
    - Cross-Entropy Loss (handles multi-class semantic discrimination)
    - Multi-Class Soft Dice Loss (handles severe remote-sensing class imbalance)
    - Multi-Class Focal Loss (focuses gradients on hard small/occluded/boundary pixels)
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from georecvit.utils.metrics import extract_boundary_mask


def multiclass_dice_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute class-averaged Soft Dice loss ignoring `ignore_index` pixels."""
    num_classes = logits.shape[1]
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0

    probs = F.softmax(logits.float(), dim=1)
    safe_targets = targets.clamp(min=0, max=num_classes - 1)
    one_hot = F.one_hot(safe_targets, num_classes=num_classes).permute(0, 3, 1, 2).float()

    valid_f = valid.unsqueeze(1).float()
    probs = probs * valid_f
    one_hot = one_hot * valid_f

    dims = (0, 2, 3)
    inter = (probs * one_hot).sum(dim=dims)
    denom = probs.sum(dim=dims) + one_hot.sum(dim=dims)
    # Average over classes present in batch (or predicted) to avoid noisy zero-support terms
    present = one_hot.sum(dim=dims) > 0
    dice = (2.0 * inter + eps) / (denom + eps)
    if present.any():
        return (1.0 - dice[present]).mean()
    return (1.0 - dice).mean()


def multiclass_focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    ignore_index: int = 255,
    gamma: float = 2.0,
) -> torch.Tensor:
    """Pixel-wise multi-class focal loss focusing on hard boundaries & small objects."""
    ce = F.cross_entropy(
        logits.float(), targets, ignore_index=ignore_index, reduction="none"
    )
    valid = targets != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    pt = torch.exp(-ce[valid]).clamp(1e-6, 1.0)
    focal = ((1.0 - pt) ** gamma) * ce[valid]
    return focal.mean()


class MultiRoundSegmentationLoss(nn.Module):
    """Outer-Loop loss with RecursiveMAS-style multi-round credit assignment."""

    def __init__(
        self,
        ignore_index: int = 255,
        non_last_loss_weight: float = 0.2,
        ce_weight: float = 1.0,
        dice_weight: float = 1.0,
        focal_weight: float = 0.5,
        active_refine_weight: float = 0.5,
        active_dice_weight: float = 1.0,
        boundary_weight: float = 0.5,
        skip_round1_loss: bool = False,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.non_last_loss_weight = non_last_loss_weight
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        self.active_refine_weight = active_refine_weight
        self.active_dice_weight = active_dice_weight
        self.boundary_weight = boundary_weight
        self.skip_round1_loss = skip_round1_loss

    def _resize_targets(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if logits.shape[-2:] != targets.shape[-2:]:
            return (
                F.interpolate(
                    targets.unsqueeze(1).float(),
                    size=logits.shape[-2:],
                    mode="nearest",
                )
                .squeeze(1)
                .long()
            )
        return targets

    def single_round_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_rs = self._resize_targets(logits, targets)

        loss = torch.zeros((), device=logits.device, dtype=torch.float32)
        if self.ce_weight > 0:
            valid = targets_rs != self.ignore_index
            if valid.any():
                loss = loss + self.ce_weight * F.cross_entropy(
                    logits.float(), targets_rs, ignore_index=self.ignore_index
                )
        if self.dice_weight > 0:
            loss = loss + self.dice_weight * multiclass_dice_loss(
                logits, targets_rs, ignore_index=self.ignore_index
            )
        if self.focal_weight > 0:
            loss = loss + self.focal_weight * multiclass_focal_loss(
                logits, targets_rs, ignore_index=self.ignore_index
            )
        return loss

    def active_window_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        active_mask_hi: torch.Tensor,
    ) -> torch.Tensor:
        """Focused CE + SoftDice + Focal + Boundary refinement loss inside routed windows."""
        targets_rs = self._resize_targets(logits, targets)
        mask_bool = active_mask_hi.squeeze(1) > 0.5
        masked_targets = torch.where(
            mask_bool, targets_rs, torch.full_like(targets_rs, self.ignore_index)
        )
        valid = masked_targets != self.ignore_index
        if not valid.any():
            return logits.sum() * 0.0
        loss = F.cross_entropy(logits.float(), masked_targets, ignore_index=self.ignore_index)
        loss = loss + 0.5 * multiclass_focal_loss(logits, masked_targets, ignore_index=self.ignore_index)
        if self.active_dice_weight > 0:
            loss = loss + self.active_dice_weight * multiclass_dice_loss(
                logits, masked_targets, ignore_index=self.ignore_index
            )
        if self.boundary_weight > 0:
            bmask = extract_boundary_mask(targets_rs, ignore_index=self.ignore_index, radius=2) & mask_bool
            if bmask.any():
                b_targets = torch.where(
                    bmask, targets_rs, torch.full_like(targets_rs, self.ignore_index)
                )
                loss = loss + self.boundary_weight * F.cross_entropy(
                    logits.float(), b_targets, ignore_index=self.ignore_index
                )
        return loss

    def forward(
        self,
        round_logits: List[torch.Tensor],
        targets: torch.Tensor,
        active_masks_hi: Optional[List[torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        per_round = [self.single_round_loss(lg, targets) for lg in round_logits]
        final_loss = per_round[-1]
        aux_candidates = per_round[1:-1] if (self.skip_round1_loss and len(per_round) > 1) else per_round[:-1]
        if aux_candidates and self.non_last_loss_weight > 0:
            aux_loss = torch.stack(aux_candidates).mean()
            total = final_loss + self.non_last_loss_weight * aux_loss
        else:
            aux_loss = torch.zeros_like(final_loss)
            total = final_loss

        active_loss = torch.zeros_like(final_loss)
        if (
            active_masks_hi
            and len(round_logits) > 1
            and self.active_refine_weight > 0
        ):
            act_terms = [
                self.active_window_loss(round_logits[i + 1], targets, m)
                for i, m in enumerate(active_masks_hi)
                if (i + 1) < len(round_logits)
            ]
            if act_terms:
                active_loss = torch.stack(act_terms).mean()
                total = total + self.active_refine_weight * active_loss

        return {
            "loss": total,
            "final_round_loss": final_loss.detach(),
            "first_round_loss": per_round[0].detach(),
            "aux_round_loss": aux_loss.detach(),
            "active_window_loss": active_loss.detach(),
        }
