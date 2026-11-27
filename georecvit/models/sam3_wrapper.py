"""Frozen SAM 3 PE-ViT + Neck + DETR + UniversalSegmentationHead wrapper.

Key capabilities verified directly against `facebookresearch/sam3`:
1. Splits the 32-layer PE-ViT cleanly at `split_block_idx=24`:
   - `forward_early_vit(images_01)`: runs `patch_embed + abs_pos + ln_pre + blocks[0:24]`
     ONCE with `torch.no_grad()`, caching `F_pe^(24) [B, 72, 72, 1024]`.
   - `forward_late_vit(x_mid, selected_indices=None, prev_pre_global=None)`:
     * Full pass (Round 1): runs frozen `blocks[24:31]` (7 window-24 layers) + `block[31]`
       (global attention) + `ln_post`.
     * Sparse recursive pass (Round r >= 2): gathers ONLY the Top-M uncertain 24x24 windows
       (`[B*M, 24, 24, 1024]`), runs them through frozen `blocks[24:31]` with exact native
       2D RoPE, scatters them back into `prev_pre_global`, then runs the single global
       `block[31]` + `ln_post` to propagate local refinements globally.
2. Runs frozen `Sam3DualViTDetNeck` (skipping scalped 0.5x scale) to obtain the 3-level
   FPN (`288x288`, `144x144`, `72x72`, `d_model=256`).
3. Runs frozen SAM 3 concept grounding (`VETextEncoder` cached class embeddings +
   6-layer `TransformerEncoderFusion` + 6-layer `TransformerDecoder` +
   `UniversalSegmentationHead`) in batched multi-class mode for all C remote-sensing
   classes simultaneously, returning:
   - `sem_logits`: `[B, C, 288, 288]` (pretrained semantic head logits)
   - `inst_probs`: `[B, C, 288, 288]` (presence-gated instance max-projection, SegEarth-OV3 style)
   - `presence_probs`: `[B, C]`
   - `decoder_context`: `[B, 256, 72, 72]` (prompt-fused visual memory averaged across active classes)
   - `fpn_feats`: list of 3 FPN maps `[288x288, 144x144, 72x72]`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as ckpt

from .uncertainty_router import gather_selected_windows, scatter_selected_windows


@dataclass
class LateViTOutput:
    """Outputs of SAM 3 PE-ViT late stage (`blocks[24:32]`)."""

    pre_global: torch.Tensor   # [B, 72, 72, 1024] output right after blocks[30] (cached for sparse rounds)
    vit_out: torch.Tensor      # [B, 1024, 72, 72] final PE-ViT output after block[31] + ln_post


@dataclass
class SAM3DecodeOutput:
    """Outputs of SAM 3 Neck + DETR + UniversalSegmentationHead."""

    sem_logits: torch.Tensor        # [B, C, 288, 288]
    inst_probs: torch.Tensor        # [B, C, 288, 288]
    presence_probs: torch.Tensor    # [B, C]
    decoder_context: torch.Tensor   # [B, 256, 72, 72]
    fpn_feats: List[torch.Tensor]   # [B, 256, 288, 288], [B, 256, 144, 144], [B, 256, 72, 72]


class SAM3RecursiveBackbone(nn.Module):
    """Wraps a frozen `Sam3Image` model for sparse closed-loop late-ViT recursion."""

    def __init__(
        self,
        sam3_model: nn.Module,
        split_block_idx: int = 24,
        resolution: int = 1008,
        use_grad_checkpoint: bool = True,
    ) -> None:
        super().__init__()
        self.sam3 = sam3_model
        self.sam3.eval()
        for p in self.sam3.parameters():
            p.requires_grad = False

        # Disable dropout inside frozen modules for deterministic gradient-checkpointed backward
        for m in self.sam3.modules():
            if isinstance(m, nn.Dropout):
                m.p = 0.0

        self.split_block_idx = split_block_idx
        self.resolution = resolution
        self.use_grad_checkpoint = use_grad_checkpoint

        self.vit = self.sam3.backbone.vision_backbone.trunk
        self.neck = self.sam3.backbone.vision_backbone
        self.scalp = int(getattr(self.sam3.backbone, "scalp", 1))
        self.embed_dim = int(self.vit.channel_list[-1])
        self.window_size = int(self.vit.blocks[split_block_idx].window_size) or 24
        self.d_model = int(self.sam3.hidden_dim)

        # Cached encoded text prompts per class tuple
        self._text_cache_key: Optional[Tuple[str, ...]] = None
        self._cached_text_out: Optional[Dict[str, torch.Tensor]] = None

    def train(self, mode: bool = True) -> "SAM3RecursiveBackbone":
        super().train(mode)
        # Keep SAM 3 strictly in eval mode so DAC/o2m training-only branches stay off
        self.sam3.eval()
        return self

    def preprocess_images(self, images_01: torch.Tensor) -> torch.Tensor:
        """Resize `[B, 3, H, W]` in `[0, 1]` to `1008x1008` and normalize to `[-1, 1]`."""
        x = F.interpolate(
            images_01.float(),
            size=(self.resolution, self.resolution),
            mode="bilinear",
            align_corners=False,
        )
        return (x - 0.5) / 0.5

    @torch.no_grad()
    def encode_class_prompts(
        self, class_prompts: Sequence[str], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        """Encode and cache all C concept text prompts using frozen SAM 3 VETextEncoder."""
        key = tuple(class_prompts)
        if self._text_cache_key == key and self._cached_text_out is not None:
            cached = self._cached_text_out
            if cached["language_features"].device == device:
                return cached

        text_out = self.sam3.backbone.forward_text(list(class_prompts), device=device)
        # Also compute normalized pooled class prototypes [C, 256] for open-vocab readout
        lang_feat = text_out["language_features"]  # [L_txt, C, 256]
        lang_mask = text_out["language_mask"]      # [C, L_txt] (True = padding)
        valid = (~lang_mask).float().transpose(0, 1).unsqueeze(-1)  # [L_txt, C, 1]
        pooled = (lang_feat * valid).sum(dim=0) / valid.sum(dim=0).clamp(min=1.0)
        text_out["class_prototypes"] = F.normalize(pooled, dim=-1)  # [C, 256]

        self._text_cache_key = key
        self._cached_text_out = text_out
        return text_out

    @torch.no_grad()
    def forward_early_vit(self, images_01: torch.Tensor) -> torch.Tensor:
        """Run frozen PE-ViT `patch_embed + pos + ln_pre + blocks[0:split_block_idx]` ONCE.

        Returns:
            `F_pe_mid`: `[B, 72, 72, 1024]` cached intermediate visual tokens.
        """
        from sam3.model.vitdet import get_abs_pos

        x = self.preprocess_images(images_01)
        param_dtype = next(self.vit.parameters()).dtype
        x = x.to(dtype=param_dtype)

        x = self.vit.patch_embed(x)  # [B, 72, 72, 1024]
        h, w = x.shape[1], x.shape[2]
        if self.vit.pos_embed is not None:
            x = x + get_abs_pos(
                self.vit.pos_embed,
                self.vit.pretrain_use_cls_token,
                (h, w),
                self.vit.retain_cls_token,
                tiling=self.vit.tile_abs_pos,
            )
        x = self.vit.ln_pre(x)

        for idx in range(self.split_block_idx):
            x = self.vit.blocks[idx](x)
        return x  # [B, 72, 72, 1024]

    def _run_block(self, blk: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_grad_checkpoint and x.requires_grad:
            return ckpt.checkpoint(blk, x, use_reentrant=False)
        return blk(x)

    def forward_late_vit(
        self,
        x_mid: torch.Tensor,
        selected_indices: Optional[torch.Tensor] = None,
        prev_pre_global: Optional[torch.Tensor] = None,
    ) -> LateViTOutput:
        """Run frozen PE-ViT `blocks[split_block_idx:32]` (full or sparse Top-M windows).

        - `blocks[24..30]` are 7 pure `window_size=24` blocks (`72/24 = 3x3 = 9` windows).
        - `blocks[31]` is the final global attention block.
        When `selected_indices` is provided (Round r >= 2), ONLY those Top-M windows
        execute `blocks[24..30]`, preserving exact 2D RoPE frequencies while skipping
        `(9 - M) / 9` of windowed compute.
        """
        num_blocks = len(self.vit.blocks)
        last_block_idx = num_blocks - 1

        if selected_indices is not None and prev_pre_global is not None:
            # Sparse window execution on blocks[split_block_idx .. last_block_idx - 1]
            sel_win, _ = gather_selected_windows(
                x_mid, selected_indices, window_size=self.window_size
            )
            h_win = sel_win
            for idx in range(self.split_block_idx, last_block_idx):
                h_win = self._run_block(self.vit.blocks[idx], h_win)
            pre_global = scatter_selected_windows(
                prev_pre_global, h_win, selected_indices, window_size=self.window_size
            )
        else:
            h = x_mid
            for idx in range(self.split_block_idx, last_block_idx):
                h = self._run_block(self.vit.blocks[idx], h)
            pre_global = h

        # Final global attention block propagates local refinements across full 72x72 tile
        x_global = self._run_block(self.vit.blocks[last_block_idx], pre_global)
        x_global = self.vit.ln_post(x_global)
        vit_out = x_global.permute(0, 3, 1, 2).contiguous()  # [B, 1024, 72, 72]
        return LateViTOutput(pre_global=pre_global, vit_out=vit_out)

    def forward_neck(self, vit_out: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """Run frozen SimpleFPN neck on `[B, 1024, 72, 72]` (skipping scalped level)."""
        num_keep = len(self.neck.convs) - self.scalp
        feats, pos_list = [], []
        for i in range(num_keep):
            conv = self.neck.convs[i]
            if self.use_grad_checkpoint and vit_out.requires_grad:
                f_out = ckpt.checkpoint(conv, vit_out, use_reentrant=False)
            else:
                f_out = conv(vit_out)
            pos_out = self.neck.position_encoding(f_out).to(f_out.dtype)
            feats.append(f_out)
            pos_list.append(pos_out)
        return feats, pos_list

    def forward_concept_decode(
        self,
        fpn_feats: List[torch.Tensor],
        fpn_pos: List[torch.Tensor],
        text_out: Dict[str, torch.Tensor],
        run_detr: bool = True,
    ) -> SAM3DecodeOutput:
        """Run frozen SAM 3 DETR + UniversalSegmentationHead across all C class prompts."""
        B = fpn_feats[0].shape[0]
        device = fpn_feats[0].device
        dtype = fpn_feats[0].dtype
        C = text_out["language_features"].shape[1]
        H_low, W_low = fpn_feats[-1].shape[-2:]
        H_hi, W_hi = fpn_feats[0].shape[-2:]

        if not run_detr:
            # Lightweight fallback mode (ablation without DETR encoder/decoder)
            proto = text_out["class_prototypes"].to(device=device, dtype=dtype)  # [C, 256]
            hi_norm = F.normalize(fpn_feats[0], dim=1)
            sem_logits = torch.einsum("bchw,kc->bkhw", hi_norm, proto) * 10.0
            inst_probs = torch.sigmoid(sem_logits)
            presence_probs = inst_probs.flatten(2).amax(dim=-1)
            return SAM3DecodeOutput(
                sem_logits=sem_logits,
                inst_probs=inst_probs,
                presence_probs=presence_probs,
                decoder_context=fpn_feats[-1],
                fpn_feats=fpn_feats,
            )

        from sam3.model.data_misc import FindStage

        # Batch all (b, c) pairs in a single forward pass: total N = B * C
        img_ids = torch.arange(B, device=device, dtype=torch.long).repeat_interleave(C)
        text_ids = torch.arange(C, device=device, dtype=torch.long).repeat(B)
        find_input = FindStage(
            img_ids=img_ids,
            text_ids=text_ids,
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )
        backbone_out = {
            "vision_features": fpn_feats[-1],
            "vision_pos_enc": fpn_pos,
            "backbone_fpn": fpn_feats,
            "language_features": text_out["language_features"],
            "language_mask": text_out["language_mask"],
            "language_embeds": text_out["language_embeds"],
        }
        geo_prompt = self.sam3._get_dummy_prompt(num_prompts=B * C)

        def _decode_fn(f0: torch.Tensor, f1: torch.Tensor, f2: torch.Tensor):
            bb_local = dict(backbone_out)
            bb_local["backbone_fpn"] = [f0, f1, f2]
            bb_local["vision_features"] = f2
            out = self.sam3.forward_grounding(
                backbone_out=bb_local,
                find_input=find_input,
                find_target=None,
                geometric_prompt=geo_prompt,
            )
            return (
                out["semantic_seg"],            # [B*C, 1, 288, 288]
                out["pred_masks"],              # [B*C, 200, 288, 288]
                out["pred_logits"],             # [B*C, 200, 1]
                out["presence_logit_dec"],      # [B*C, 1]
                out["encoder_hidden_states"],   # [72*72, B*C, 256]
            )

        if self.use_grad_checkpoint and any(f.requires_grad for f in fpn_feats):
            sem_seg, pred_masks, pred_logits, pres_logit, enc_states = ckpt.checkpoint(
                _decode_fn, fpn_feats[0], fpn_feats[1], fpn_feats[2], use_reentrant=False
            )
        else:
            sem_seg, pred_masks, pred_logits, pres_logit, enc_states = _decode_fn(
                fpn_feats[0], fpn_feats[1], fpn_feats[2]
            )

        # Reshape semantic head output: [B, C, 288, 288]
        sem_logits = sem_seg.squeeze(1).view(B, C, H_hi, W_hi)

        # Compute SegEarth-OV3-style presence-gated instance probability map: [B, C, 288, 288]
        presence_probs = pres_logit.sigmoid().view(B, C)  # [B, C]
        q_scores = pred_logits.sigmoid().view(B, C, -1) * presence_probs.unsqueeze(-1)
        # Soft-weighted top-K instance aggregation (differentiable & memory-friendly)
        top_k = min(16, q_scores.shape[-1])
        top_scores, top_idx = torch.topk(q_scores, k=top_k, dim=-1)  # [B, C, K]
        masks_bc = pred_masks.view(B, C, -1, H_hi, W_hi)
        gather_m_idx = top_idx[:, :, :, None, None].expand(B, C, top_k, H_hi, W_hi)
        top_masks = torch.gather(masks_bc, dim=2, index=gather_m_idx).sigmoid()
        inst_probs = (top_masks * top_scores[:, :, :, None, None]).amax(dim=2)  # [B, C, 288, 288]

        # Aggregate prompt-conditioned encoder hidden states weighted by class presence -> [B, 256, 72, 72]
        enc_map = (
            enc_states.permute(1, 2, 0)
            .contiguous()
            .view(B, C, self.d_model, H_low, W_low)
        )
        w_pres = F.softmax(pres_logit.view(B, C), dim=1)[:, :, None, None, None]
        decoder_context = (enc_map * w_pres).sum(dim=1)  # [B, 256, 72, 72]

        return SAM3DecodeOutput(
            sem_logits=sem_logits,
            inst_probs=inst_probs,
            presence_probs=presence_probs,
            decoder_context=decoder_context,
            fpn_feats=fpn_feats,
        )
