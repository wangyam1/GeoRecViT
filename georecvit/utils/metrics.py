"""Fast confusion-matrix evaluation metrics for remote sensing segmentation."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch


class SegmentationMeter:
    """Computes mIoU, per-class IoU, mF1 (Dice), and Overall Accuracy (OA)."""

    def __init__(
        self,
        num_classes: int,
        class_names: Optional[Sequence[str]] = None,
        ignore_index: int = 255,
    ) -> None:
        self.num_classes = num_classes
        self.class_names = (
            list(class_names) if class_names is not None else [f"class_{i}" for i in range(num_classes)]
        )
        self.ignore_index = ignore_index
        self.reset()

    def reset(self) -> None:
        self.conf_mat = torch.zeros((self.num_classes, self.num_classes), dtype=torch.int64)

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update confusion matrix with integer class predictions `[B, H, W]` and targets `[B, H, W]`."""
        p = preds.detach().reshape(-1).to(dtype=torch.int64, device="cpu")
        t = targets.detach().reshape(-1).to(dtype=torch.int64, device="cpu")
        valid = (t != self.ignore_index) & (t >= 0) & (t < self.num_classes)
        if not valid.any():
            return
        idx = t[valid] * self.num_classes + p[valid].clamp(0, self.num_classes - 1)
        bins = torch.bincount(idx, minlength=self.num_classes * self.num_classes)
        self.conf_mat += bins.view(self.num_classes, self.num_classes)

    def compute(self) -> Dict[str, object]:
        cm = self.conf_mat.double()
        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp
        denom_iou = tp + fp + fn
        valid_cls = denom_iou > 0

        iou = torch.where(valid_cls, tp / denom_iou.clamp(min=1.0), torch.zeros_like(tp))
        f1 = torch.where(
            valid_cls,
            (2.0 * tp) / (2.0 * tp + fp + fn).clamp(min=1.0),
            torch.zeros_like(tp),
        )
        miou = float(iou[valid_cls].mean().item() * 100.0) if valid_cls.any() else 0.0
        mf1 = float(f1[valid_cls].mean().item() * 100.0) if valid_cls.any() else 0.0
        oa = float((tp.sum() / cm.sum().clamp(min=1.0)).item() * 100.0)

        per_class_iou = {
            name: round(float(iou[i].item() * 100.0), 2)
            for i, name in enumerate(self.class_names)
        }
        return {
            "mIoU": round(miou, 2),
            "mF1": round(mf1, 2),
            "OA": round(oa, 2),
            "per_class_IoU": per_class_iou,
        }
