"""Frozen Remote-Sensing Structural ViT Expert (DINOv3-SAT-493M / DINOv3-LVD / DINOv2).

Loads Meta's satellite-pretrained `facebook/dinov3-vitl16-pretrain-sat493m`
(or natural-image `dinov3-vitl16-pretrain-lvd1689m` / `dinov2-with-registers-large`
for controlled ablations) from local ModelScope or HuggingFace paths, keeps all
parameters strictly frozen, and supports BOTH:
  - Single-pass extraction (`forward`), and
  - Split Early/Late recursive execution (`forward_early` on layers `[0:split_layer_idx]`
    cached ONCE per image + differentiable `forward_late` on layers `[split_layer_idx:24]`
    with exact 2D RoPE) enabling true bidirectional `A <-> B` and `B <-> B` recursion!
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


@dataclass
class DINOv3EarlyOutput:
    """Cached early-trunk state of frozen DINOv3-SAT (`layers[0:split_layer_idx]`)."""

    prefix_tokens: torch.Tensor              # [B, prefix, C] (CLS + registers)
    spatial_mid: torch.Tensor                # [B, h_p, w_p, C]
    position_embeddings: object              # 2D RoPE (cos, sin) tuple or None
    h_p: int
    w_p: int


class DINOv3Expert(nn.Module):
    """Frozen DINOv3 / DINOv2 ViT structural feature extractor with split early/late recursion."""

    DEFAULT_MEAN = (0.485, 0.456, 0.406)
    DEFAULT_STD = (0.229, 0.224, 0.225)

    def __init__(
        self,
        model_path: str,
        input_size: int = 512,
        target_grid_hw: Tuple[int, int] = (72, 72),
        split_layer_idx: int = 18,
        use_grad_checkpoint: bool = True,
        dtype: Optional[torch.dtype] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        from transformers import AutoImageProcessor, AutoModel

        self.model_path = model_path
        self.input_size = input_size
        self.target_grid_hw = target_grid_hw
        self.use_grad_checkpoint = use_grad_checkpoint

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
        self.num_layers = int(getattr(cfg, "num_hidden_layers", 24))
        self.split_layer_idx = min(max(1, int(split_layer_idx)), max(1, self.num_layers - 1))

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

    def _get_encoder_layers(self) -> Optional[nn.ModuleList]:
        if hasattr(self.backbone, "model") and hasattr(self.backbone.model, "layer"):
            return self.backbone.model.layer
        if hasattr(self.backbone, "encoder") and hasattr(self.backbone.encoder, "layer"):
            return self.backbone.encoder.layer
        return None

    def _tokens_to_grid(self, tokens: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:
        B, N, C = tokens.shape
        expected_spatial = h_p * w_p
        prefix = N - expected_spatial
        if prefix < 0:
            side = int(math.sqrt(N))
            spatial_tokens = tokens[:, -side * side :, :]
            h_p = w_p = side
        else:
            spatial_tokens = tokens[:, prefix:, :]
        feat_map = spatial_tokens.view(B, h_p, w_p, C).permute(0, 3, 1, 2).contiguous()
        if (h_p, w_p) != self.target_grid_hw:
            feat_map = F.interpolate(
                feat_map.float(),
                size=self.target_grid_hw,
                mode="bilinear",
                align_corners=False,
            ).to(tokens.dtype)
        return feat_map.permute(0, 2, 3, 1).contiguous()

    @torch.no_grad()
    def forward_early(self, images_01: torch.Tensor) -> DINOv3EarlyOutput:
        """Run frozen early DINOv3 layers `[0:split_layer_idx]` ONCE per image."""
        pixel_values = self._preprocess(images_01)
        h_p = pixel_values.shape[-2] // self.patch_size
        w_p = pixel_values.shape[-1] // self.patch_size

        layers = self._get_encoder_layers()
        if layers is None or not hasattr(self.backbone, "embeddings"):
            raise RuntimeError("Backbone does not expose split DINOv3ViT layers.")

        hidden_states = self.backbone.embeddings(pixel_values)
        pos_emb = (
            self.backbone.rope_embeddings(pixel_values)
            if hasattr(self.backbone, "rope_embeddings")
            else None
        )
        for layer_module in layers[: self.split_layer_idx]:
            if pos_emb is not None:
                hidden_states = layer_module(hidden_states, position_embeddings=pos_emb)
            else:
                out = layer_module(hidden_states)
                hidden_states = out[0] if isinstance(out, tuple) else out

        B, N, C = hidden_states.shape
        prefix = max(0, N - h_p * w_p)
        prefix_tokens = hidden_states[:, :prefix, :]
        spatial_mid = hidden_states[:, prefix:, :].view(B, h_p, w_p, C).contiguous()
        return DINOv3EarlyOutput(
            prefix_tokens=prefix_tokens,
            spatial_mid=spatial_mid,
            position_embeddings=pos_emb,
            h_p=h_p,
            w_p=w_p,
        )

    def forward_late(
        self,
        early_out: DINOv3EarlyOutput,
        spatial_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run frozen late DINOv3 layers `[split_layer_idx:24]` with exact 2D RoPE.

        Differentiable w.r.t. `spatial_delta` so top-down `A -> B` and `B -> B`
        feedback links receive end-to-end gradients while DINOv3 weights stay frozen!
        """
        layers = self._get_encoder_layers()
        assert layers is not None

        spatial = early_out.spatial_mid
        if spatial_delta is not None:
            if spatial_delta.shape[1:3] != (early_out.h_p, early_out.w_p):
                spatial_delta = (
                    F.interpolate(
                        spatial_delta.permute(0, 3, 1, 2).float(),
                        size=(early_out.h_p, early_out.w_p),
                        mode="bilinear",
                        align_corners=False,
                    )
                    .permute(0, 2, 3, 1)
                    .to(spatial.dtype)
                )
            spatial = spatial + spatial_delta.to(spatial.dtype)

        B, h_p, w_p, C = spatial.shape
        hidden_states = torch.cat(
            [early_out.prefix_tokens.to(spatial.dtype), spatial.reshape(B, h_p * w_p, C)],
            dim=1,
        )
        pos_emb = early_out.position_embeddings
        use_ckpt = self.use_grad_checkpoint and torch.is_grad_enabled() and hidden_states.requires_grad

        for layer_module in layers[self.split_layer_idx :]:
            if pos_emb is not None:
                if use_ckpt:
                    hidden_states = checkpoint(
                        lambda h, lm=layer_module: lm(h, position_embeddings=pos_emb),
                        hidden_states,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = layer_module(hidden_states, position_embeddings=pos_emb)
            else:
                out = layer_module(hidden_states)
                hidden_states = out[0] if isinstance(out, tuple) else out

        if hasattr(self.backbone, "norm"):
            hidden_states = self.backbone.norm(hidden_states)
        elif hasattr(self.backbone, "layernorm"):
            hidden_states = self.backbone.layernorm(hidden_states)
        return self._tokens_to_grid(hidden_states, h_p, w_p)

    @torch.no_grad()
    def forward(self, images_01: torch.Tensor) -> torch.Tensor:
        """Extract frozen structural features aligned to `target_grid_hw`: `[B, 72, 72, C]`."""
        pixel_values = self._preprocess(images_01)
        outputs = self.backbone(pixel_values=pixel_values)
        tokens = outputs.last_hidden_state
        h_p = pixel_values.shape[-2] // self.patch_size
        w_p = pixel_values.shape[-1] // self.patch_size
        return self._tokens_to_grid(tokens, h_p, w_p)
