"""Remote Sensing Semantic Segmentation Datasets (LoveDA & OpenEarthMap).

Supports loading directly from HuggingFace Datasets (`chloechia/loveda`,
`EVER-Z/open_earth_map`) or local disk caches on Colab (`/content/data/...`).
Automatically normalizes 1-indexed official masks (where 0 = no-data) or
0-indexed masks into contiguous `[0, num_classes - 1]` + `ignore_index = 255`.
"""

from __future__ import annotations

from pathlib import Path
import random
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import yaml


def load_yaml_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def detect_label_offset(
    hf_dataset: Any,
    mask_column: str,
    num_classes: int,
    probe_samples: int = 32,
) -> int:
    """Detect whether raw masks are 1-indexed (0=no-data, 1..C) or 0-indexed (0..C-1)."""
    max_non_ignore = -1
    n = min(len(hf_dataset), probe_samples)
    for idx in range(n):
        mask = hf_dataset[idx][mask_column]
        arr = np.asarray(mask)
        if arr.ndim == 3:
            arr = arr[..., 0]
        valid = arr[arr < 250]
        if valid.size > 0:
            max_non_ignore = max(max_non_ignore, int(valid.max()))
    # Official LoveDA has values 0..7 (C=7); Official OpenEarthMap has 0..8 (C=8).
    if max_non_ignore == num_classes:
        return 1
    return 0


def normalize_mask_array(
    arr: np.ndarray,
    num_classes: int,
    label_offset: int,
    ignore_index: int = 255,
) -> np.ndarray:
    """Map raw integer mask to contiguous `[0, num_classes - 1]` with `ignore_index`."""
    if arr.ndim == 3:
        arr = arr[..., 0]
    raw = arr.astype(np.int64)
    out = np.full_like(raw, ignore_index)
    if label_offset == 1:
        # 0 is no-data (ignore), 1..C -> 0..C-1
        valid = (raw >= 1) & (raw <= num_classes)
        out[valid] = raw[valid] - 1
    else:
        valid = (raw >= 0) & (raw < num_classes)
        out[valid] = raw[valid]
    return out


class RemoteSensingSegDataset(Dataset):
    """PyTorch Dataset wrapper for HuggingFace Remote Sensing segmentation splits."""

    def __init__(
        self,
        hf_split: Any,
        image_column: str,
        mask_column: str,
        num_classes: int,
        ignore_index: int = 255,
        label_mode: str = "auto",
        target_size: int = 1008,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        if max_samples is not None and max_samples > 0 and max_samples < len(hf_split):
            # Deterministic uniform striding covers all geographic regions / domains evenly
            indices = np.linspace(0, len(hf_split) - 1, max_samples, dtype=int).tolist()
            hf_split = hf_split.select(indices)
        self.ds = hf_split
        self.image_column = image_column
        self.mask_column = mask_column
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.target_size = target_size
        self.augment = augment

        if label_mode == "auto":
            self.label_offset = detect_label_offset(self.ds, mask_column, num_classes)
        elif label_mode in {"1_indexed", "one_indexed", "official"}:
            self.label_offset = 1
        elif label_mode in {"0_indexed", "zero_indexed"}:
            self.label_offset = 0
        else:
            raise ValueError(f"Unsupported label_mode: {label_mode}")

    def __len__(self) -> int:
        return len(self.ds)

    def _apply_augment(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        # Overhead remote-sensing imagery is rotation & flip invariant (D4 symmetry group)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k == 1:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        elif k == 2:
            img = img.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)
        elif k == 3:
            img = img.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)
        return img, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.ds[idx]
        img = row[self.image_column]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.asarray(img))
        img = img.convert("RGB")

        raw_mask = row[self.mask_column]
        mask_arr = normalize_mask_array(
            np.asarray(raw_mask),
            num_classes=self.num_classes,
            label_offset=self.label_offset,
            ignore_index=self.ignore_index,
        )
        mask_img = Image.fromarray(mask_arr.astype(np.uint8))

        if self.augment:
            img, mask_img = self._apply_augment(img, mask_img)

        if img.size != (self.target_size, self.target_size):
            img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            mask_img = mask_img.resize((self.target_size, self.target_size), Image.NEAREST)

        # Image tensor in [0, 1] float32 [3, H, W]; each backbone applies its own normalization
        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(np.asarray(mask_img, dtype=np.int64))
        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "index": idx,
        }


class LocalPairedSegDataset(Dataset):
    """Dataset for official disk-stored paired PNGs (e.g. LoveDA Zenodo Urban/Rural)."""

    def __init__(
        self,
        pairs: Sequence[Tuple[Path, Path]],
        num_classes: int,
        ignore_index: int = 255,
        label_mode: str = "auto",
        target_size: int = 1008,
        augment: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        pair_list = list(pairs)
        if max_samples is not None and max_samples > 0 and max_samples < len(pair_list):
            # Deterministic uniform striding preserves balanced Urban + Rural coverage
            indices = np.linspace(0, len(pair_list) - 1, max_samples, dtype=int).tolist()
            pair_list = [pair_list[i] for i in indices]
        self.pairs = pair_list
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.target_size = target_size
        self.augment = augment

        if label_mode == "auto":
            max_v = -1
            for _, mp in self.pairs[: min(32, len(self.pairs))]:
                arr = np.asarray(Image.open(mp))
                if arr.ndim == 3:
                    arr = arr[..., 0]
                valid = arr[arr < 250]
                if valid.size > 0:
                    max_v = max(max_v, int(valid.max()))
            self.label_offset = 1 if max_v == num_classes else 0
        elif label_mode in {"1_indexed", "one_indexed", "official"}:
            self.label_offset = 1
        else:
            self.label_offset = 0

    def __len__(self) -> int:
        return len(self.pairs)

    def _apply_augment(
        self, img: Image.Image, mask: Image.Image
    ) -> Tuple[Image.Image, Image.Image]:
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
        k = random.randint(0, 3)
        if k == 1:
            img = img.transpose(Image.ROTATE_90)
            mask = mask.transpose(Image.ROTATE_90)
        elif k == 2:
            img = img.transpose(Image.ROTATE_180)
            mask = mask.transpose(Image.ROTATE_180)
        elif k == 3:
            img = img.transpose(Image.ROTATE_270)
            mask = mask.transpose(Image.ROTATE_270)
        return img, mask

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_path, mask_path = self.pairs[idx]
        img = Image.open(img_path).convert("RGB")
        mask_arr = normalize_mask_array(
            np.asarray(Image.open(mask_path)),
            num_classes=self.num_classes,
            label_offset=self.label_offset,
            ignore_index=self.ignore_index,
        )
        mask_img = Image.fromarray(mask_arr.astype(np.uint8))

        if self.augment:
            img, mask_img = self._apply_augment(img, mask_img)
        if img.size != (self.target_size, self.target_size):
            img = img.resize((self.target_size, self.target_size), Image.BILINEAR)
            mask_img = mask_img.resize((self.target_size, self.target_size), Image.NEAREST)

        img_np = np.asarray(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).contiguous()
        mask_tensor = torch.from_numpy(np.asarray(mask_img, dtype=np.int64))
        return {"image": img_tensor, "mask": mask_tensor, "index": idx}


def _collect_loveda_pairs(root: Path, split: str, domains: Sequence[str]) -> list[Tuple[Path, Path]]:
    pairs: list[Tuple[Path, Path]] = []
    for dom in domains:
        img_dir = root / split / dom / "images_png"
        msk_dir = root / split / dom / "masks_png"
        if not img_dir.is_dir() or not msk_dir.is_dir():
            continue
        for img_p in sorted(img_dir.glob("*.png")):
            msk_p = msk_dir / img_p.name
            if msk_p.is_file():
                pairs.append((img_p, msk_p))
    return pairs


def build_rs_datasets(
    cfg: Dict[str, Any],
    cache_dir: Optional[str] = None,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
) -> Tuple[Dataset, Dataset]:
    """Load train and val splits from local LoveDA Zenodo folders or HuggingFace Hub."""
    ds_cfg = cfg["dataset"]
    source = str(ds_cfg.get("source", "hf")).lower()
    num_classes = int(ds_cfg["num_classes"])
    ignore_index = int(ds_cfg.get("ignore_index", 255))
    label_mode = str(ds_cfg.get("label_mode", "auto"))
    target_size = int(ds_cfg.get("sam3_resolution", 1008))

    if source == "local_loveda":
        root = Path(ds_cfg.get("local_root") or (Path(cache_dir or "/content/georecvit_cache/datasets") / "loveda"))
        domains = ds_cfg.get("domains", ["Urban", "Rural"])
        train_pairs = _collect_loveda_pairs(root, str(ds_cfg.get("train_split", "Train")), domains)
        val_pairs = _collect_loveda_pairs(root, str(ds_cfg.get("val_split", "Val")), domains)
        if not val_pairs:
            raise FileNotFoundError(
                f"LoveDA Val pairs not found under {root}. Run setup_colab_env.py to fetch Zenodo Val.zip."
            )
        # Allow val-only mode (e.g. Baseline 0 eval) when Train.zip hasn't been extracted yet
        if not train_pairs:
            train_pairs = val_pairs[:4]
        train_ds = LocalPairedSegDataset(
            train_pairs, num_classes=num_classes, ignore_index=ignore_index,
            label_mode=label_mode, target_size=target_size, augment=True, max_samples=max_train_samples,
        )
        val_ds = LocalPairedSegDataset(
            val_pairs, num_classes=num_classes, ignore_index=ignore_index,
            label_mode=label_mode, target_size=target_size, augment=False, max_samples=max_val_samples,
        )
        return train_ds, val_ds

    from datasets import load_dataset

    hf_repo = ds_cfg["hf_repo"]
    train_hf = load_dataset(hf_repo, split=ds_cfg["train_split"], cache_dir=cache_dir)
    val_hf = load_dataset(hf_repo, split=ds_cfg["val_split"], cache_dir=cache_dir)

    common = dict(
        image_column=ds_cfg["image_column"],
        mask_column=ds_cfg["mask_column"],
        num_classes=num_classes,
        ignore_index=ignore_index,
        label_mode=label_mode,
        target_size=target_size,
    )
    train_ds = RemoteSensingSegDataset(
        train_hf, augment=True, max_samples=max_train_samples, **common
    )
    val_ds = RemoteSensingSegDataset(
        val_hf, augment=False, max_samples=max_val_samples, **common
    )
    return train_ds, val_ds
