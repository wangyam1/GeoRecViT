"""Frozen Remote-Sensing Structural ViT Expert (DINOv3-SAT-493M / DINOv3-LVD / DINOv2).

Loads Meta's satellite-pretrained `facebook/dinov3-vitl16-pretrain-sat493m`
(or natural-image `dinov3-vitl16-pretrain-lvd1689m` / `dinov2-with-registers-large`
for controlled ablations) from local ModelScope or HuggingFace paths, keeps all
parameters strictly frozen, and extracts dense spatial patch grids aligned to
SAM 3's 72x72 ViT grid.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOv3Expert(nn.Module):
    """Frozen DINOv3 / DINOv2 ViT structural feature extractor."""

    # Standard ImageNet / SAT-493M normalization stats (overridden if preprocessor config present)
    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_path: str,
        input_size: int = 512,
        target_grid_hw: Tuple[int, int] = (72, 72),
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.model_path = model_path
        self.input_size = input_size
        self.target_grid_hw = target_grid_hw

        # Load normalization parameters if available in checkpoint folder
        mean, std = self.DEFAULT_MEAN, self.DEFAULT_STD
        try:
            proc = AutoImageProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code
            )
            if getattr(proc, "image_mean", None) is not None:
                mean = tuple(float(v) for v in proc.image_mean)
            if getattr(proc, "image_std", None) is not None:
                std = tuple(float(v) for v in proc.image_std)
        except Exception:
            pass

        self.register_buffer(
            "pixel_mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "pixel_std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        )

        load_kwargs = {"trust_remote_code": trust_remote_code}
        if dtype is not None:
            load_kwargs["torch_dtype"] = dtype
        self.backbone = AutoModel.from_pretrained(model_path, **load_kwargs)
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        cfg = self.backbone.config
        self.embed_dim = int(getattr(cfg, "hidden_size", 1024))
        self.patch_size = int(getattr(cfg, "patch_size", 16))
        self.num_register_tokens = int(getattr(cfg, "num_register_tokens", 4))

    def train(self, mode: bool = True) -> "DINOv3Expert":
        super().train(mode)
        self.backbone.eval()
        return self

    def _preprocess(self, images_01: torch.Tensor) -> torch.Tensor:
        """Resize RGB images in `[0, 1]` to `input_size` (divisible by patch_size) and normalize."""
        ps = self.patch_size
        sz = (self.input_size // ps) * ps
        x = F.interpolate(
            images_01.float(), size=(sz, sz), mode="bilinear", align_corners=False
        )
        x = (x - self.pixel_mean.to(x.device)) / self.pixel_std.to(x.device)
        param_dtype = next(self.backbone.parameters()).dtype
        return x.to(dtype=param_dtype)

    @torch.no_grad()
    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        """Extract frozen structural features aligned to `target_grid_hw`: `[B, 72, 72, C]`."""
        pixel_values = self._preprocess(images_01)
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state  # [B, 1 + num_reg + H_p*W_p, C]

        B, N, C = tokens.shape
        h_p = pixel_values.shape[-2] // self.patch_size
        w_p = pixel_values.shape[-1] // self.patch_size
        expected_spatial = h_p * w_p

        prefix = N - expected_spatial
        if prefix < 0:
            # Fallback if model uses different internal pooling / patch count
            side = int(math.sqrt(N))
            spatial_tokens = tokens[:, -side * side :, :]
            h_p = w_p = side
        else:
            spatial_tokens = tokens[:, prefix:, :]

        feat_map = (
            spatial_tokens.view(B, h_p, w_p, C).permute(0, 3, 1, 2).contiguous()
        )
        if (h_p, w_p) != self.target_grid_hw:
            feat_map = F.interpolate(
                feat_map.float(),
                size=self.target_grid_hw,
                mode="bilinear",
                align_corners=False,
            ).to(tokens.dtype)
        return feat_map.permute(0, 2, 3, 1).contiguous()  # [B, 72, 72, C]
