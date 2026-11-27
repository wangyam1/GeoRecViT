"""Lightweight checkpoint helpers saving ONLY trainable RecursiveLink modules (~30-60MB)."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_georecvit_links(
    model: Any,
    output_dir: str | Path,
    tag: str = "best",
    extra_meta: Optional[Dict[str, Any]] = None,
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state = {
        name: mod.state_dict() for name, mod in model.trainable_modules().items()
    }
    ckpt_path = out_dir / f"georecvit_links_{tag}.pt"
    torch.save(state, ckpt_path)

    meta: Dict[str, Any] = {
        "config": asdict(model.cfg),
        "param_counts": model.count_parameters(),
        "modules": sorted(state.keys()),
    }
    if extra_meta:
        meta["extra"] = extra_meta
    with open(out_dir / f"georecvit_links_{tag}.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)
    return ckpt_path


def load_georecvit_links(
    model: Any,
    ckpt_path: str | Path,
    strict: bool = False,
) -> Dict[str, Any]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    loaded = {}
    mods = model.trainable_modules()
    for name, mod in mods.items():
        if name in state:
            res = mod.load_state_dict(state[name], strict=strict)
            loaded[name] = {
                "missing": list(res.missing_keys),
                "unexpected": list(res.unexpected_keys),
            }
    return loaded
