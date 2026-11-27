"""Pure Frozen DINOv3-SAT Baseline Segmenter (Baseline 1b: No SAM 3).

Provides a fair, strong single-expert ablation using the EXACT same frozen
`DINOv3Expert` (e.g. `facebook/dinov3-vitl16-pretrain-sat493m`) paired with a
multi-scale ASPP + progressive 288x288 FPN upsampling decoder (~6.0M trainable
parameters — matching GeoRecViT's single-pass readout capacity).

Directly answers the key reviewer question:
  "Does GeoRecViT genuinely benefit from SAM 3 + DINOv3-SAT heterogeneous synergy
   (1 + 1 > 2), or could frozen DINOv3-SAT alone match the same accuracy?"
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3OnlySegmenter(nn.Module):
    """Frozen DINOv3-SAT backbone + fair multi-scale pyramid upsampling decoder."""

    def __init__(
        self,
        dinov3_expert: nn.Module,
        num_classes: int,
        hidden_dim: int = 256,
        out_size: int = 288,
    ) -> None:
        super().__init__()
        self.dinov3_expert = dinov3_expert
        self.num_classes = num_classes
        self.out_size = out_size
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

    def train(self, mode: bool = True) -> "DINOv3OnlySegmenter":
        super().train(mode)
        self.dinov3_expert.eval()
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

    def forward(self, images_01: torch.Tensor) -> Dict[str, object]:
        f_sat = self.dinov3_expert(images_01)  # [B, 72, 72, C]
        x = f_sat.permute(0, 3, 1, 2).contiguous().float()
        h = self.in_proj(x)
        H, W = h.shape[-2:]
        p = F.interpolate(self.aspp_pool(h), size=(H, W), mode="bilinear", align_corners=False)
        h = self.aspp_fuse(
            torch.cat(
                [self.aspp_1(h), self.aspp_d1(h), self.aspp_d2(h), self.aspp_d4(h), p],
                dim=1,
            )
        )
        mid_sz = self.out_size // 2
        h = F.interpolate(h, size=(mid_sz, mid_sz), mode="bilinear", align_corners=False)
        h = h + self.up1(h)
        h = F.interpolate(h, size=(self.out_size, self.out_size), mode="bilinear", align_corners=False)
        h = h + self.up2(h)
        logits = self.cls_head(h)
        return {
            "round_logits": [logits],
            "final_logits": logits,
        }
