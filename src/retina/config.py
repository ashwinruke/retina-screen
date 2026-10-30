"""YAML config loading with defaults and command-line overrides."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Iterable

import yaml

DEFAULTS: dict[str, Any] = {
    "experiment": {
        "name": "run",
        "out_dir": "outputs",
    },
    "data": {
        # Cached manifest written by scripts/cache_images.py
        "manifest": None,
        "train_splits": ["train"],
        "val_splits": ["val"],
        # Optional filter on the manifest's `dataset` column, e.g. ["aptos"]
        "datasets": None,
    },
    "preprocess": {
        "size": 512,
        "ben_graham": True,
        "mask_scale": 0.95,
    },
    "model": {
        "backbone": "efficientnet_b3",
        "head": "regression",  # regression | classification
        "num_classes": 5,
        "pretrained": True,
        "dropout": 0.3,
        "drop_path": 0.2,
    },
    "train": {
        "image_size": 512,
        "epochs": 10,
        "batch_size": 16,
        "lr": 1e-3,  # head learning rate
        "backbone_lr_mult": 0.3,  # backbone lr = lr * this
        "weight_decay": 1e-4,
        "warmup_epochs": 1,
        "freeze_epochs": 0,  # train only the head for the first N epochs
        "sampler_power": 0.5,  # 0 = no rebalancing; 1 = fully balanced classes
        "label_smoothing": 0.05,  # classification head only
        "grad_clip": 1.0,
        "amp": True,
        "channels_last": True,  # faster fp16 convs on GPU; ignored on CPU
        "num_workers": 4,
        "seed": 42,
        "init_from": None,  # checkpoint to start from (e.g. stage 1 best)
        "data_parallel": True,  # use all visible GPUs (Kaggle T4 x2)
        "max_steps_per_epoch": None,  # for smoke tests
        "tta": True,  # flip TTA during validation
    },
}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _parse_value(raw: str) -> Any:
    """Parse an override value with YAML rules (so 1e-4, true, [a,b] work)."""
    return yaml.safe_load(raw)


def apply_overrides(cfg: dict, overrides: Iterable[str]) -> dict:
    """Apply ``section.key=value`` overrides, e.g. ``train.epochs=3``."""
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Override must look like key.sub=value, got: {item}")
        dotted, raw = item.split("=", 1)
        node = cfg
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = _parse_value(raw)
    return cfg


def load_config(path: str | Path | None, overrides: Iterable[str] = ()) -> dict:
    cfg = copy.deepcopy(DEFAULTS)
    if path is not None:
        with open(path) as f:
            user = yaml.safe_load(f) or {}
        cfg = deep_merge(cfg, user)
    return apply_overrides(cfg, overrides)


def save_config(cfg: dict, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
