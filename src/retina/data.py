"""PyTorch dataset, augmentations and loaders over cached manifests."""
from __future__ import annotations

from typing import Sequence

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from .preprocess import read_rgb

_ALB_V2 = int(A.__version__.split(".")[0]) >= 2


def _affine(background: int) -> A.BasicTransform:
    # Fundus images have no fixed orientation, so any rotation is valid.
    common = dict(scale=(0.9, 1.1), rotate=(-180, 180), p=0.9)
    if _ALB_V2:
        return A.Affine(border_mode=cv2.BORDER_CONSTANT, fill=background, **common)
    return A.Affine(mode=cv2.BORDER_CONSTANT, cval=background, **common)


def build_transforms(image_size: int, train: bool, background: int = 128) -> A.Compose:
    ops: list = [A.Resize(image_size, image_size)]
    if train:
        ops += [
            _affine(background),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
        ]
    ops += [A.Normalize(), ToTensorV2()]
    return A.Compose(ops)


class FundusDataset(Dataset):
    """Reads already-preprocessed (cached) images listed in a manifest."""

    def __init__(self, df: pd.DataFrame, transform: A.Compose):
        self.paths = df["image_path"].tolist()
        self.grades = df["grade"].astype(int).to_numpy()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        img = read_rgb(self.paths[i])
        x = self.transform(image=img)["image"]
        return x, torch.tensor(self.grades[i], dtype=torch.long)


def select(
    df: pd.DataFrame, splits: Sequence[str] | None, datasets: Sequence[str] | None = None
) -> pd.DataFrame:
    """Filter a manifest by split (["all"] or None keeps every split) and dataset."""
    out = df
    if splits and "all" not in splits:
        out = out[out["split"].isin(splits)]
    if datasets:
        out = out[out["dataset"].isin(datasets)]
    return out.reset_index(drop=True)


def class_balanced_sampler(grades: np.ndarray, power: float) -> WeightedRandomSampler | None:
    """Sample each image with weight 1 / count(grade) ** power.

    power=0 leaves the natural distribution, power=1 fully balances classes.
    0.5 is a middle ground that helps rare grades without heavy oversampling.
    """
    if power <= 0:
        return None
    counts = np.bincount(grades, minlength=5).astype(float)
    weights = 1.0 / np.power(np.maximum(counts[grades], 1.0), power)
    return WeightedRandomSampler(
        torch.as_tensor(weights, dtype=torch.double), num_samples=len(grades), replacement=True
    )


def make_loader(
    df: pd.DataFrame,
    image_size: int,
    batch_size: int,
    train: bool,
    num_workers: int = 4,
    background: int = 128,
    sampler_power: float = 0.0,
) -> DataLoader:
    ds = FundusDataset(df, build_transforms(image_size, train=train, background=background))
    sampler = class_balanced_sampler(ds.grades, sampler_power) if train else None
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=train and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train and len(ds) > batch_size,
        persistent_workers=num_workers > 0,
    )
