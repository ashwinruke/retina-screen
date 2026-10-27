"""Fundus image preprocessing.

The same functions run when caching the training set and inside the demo, and
the settings travel with every checkpoint, so training and inference always
see identically prepared images.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Background value outside the retina after Ben Graham normalisation. 128 is
# the "no local contrast" level, so it carries no signal for the network.
BEN_BACKGROUND = 128


def read_rgb(path: str | Path) -> np.ndarray:
    """Read an image file as an RGB uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_to_fundus(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """Remove the black border around the retina.

    Rows/columns whose brightest pixel stays under ``threshold`` are dropped.
    If almost nothing is brighter than the threshold (a nearly black,
    ungradable image), the image is returned unchanged.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > threshold
    if mask.mean() < 0.01:
        return img
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return img[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def pad_to_square(img: np.ndarray, value: int = 0) -> np.ndarray:
    """Centre the image on a square canvas.

    Some fundus photos clip the top and bottom of the retina; padding (rather
    than stretching) keeps the circle round.
    """
    h, w = img.shape[:2]
    side = max(h, w)
    out = np.full((side, side, 3), value, dtype=img.dtype)
    y, x = (side - h) // 2, (side - w) // 2
    out[y : y + h, x : x + w] = img
    return out


def ben_graham(img: np.ndarray, sigma_frac: float = 1 / 30) -> np.ndarray:
    """Subtract local average colour: 4*img - 4*blur(img) + 128.

    Removes slow illumination differences between cameras and makes small
    lesions stand out.
    """
    sigma = img.shape[0] * sigma_frac
    blur = cv2.GaussianBlur(img, (0, 0), sigma)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def circular_mask(img: np.ndarray, scale: float = 0.95, fill: int = 0) -> np.ndarray:
    """Set everything outside a centred circle to ``fill``."""
    side = img.shape[0]
    mask = np.zeros((side, side), np.uint8)
    cv2.circle(mask, (side // 2, side // 2), int(side / 2 * scale), 1, thickness=-1)
    out = img.copy()
    out[mask == 0] = fill
    return out


def fundus_mask(img: np.ndarray, threshold: int = 10) -> np.ndarray:
    """uint8 mask (1 = retina, 0 = black background or padding).

    Small dark holes inside the retina (vessels, haemorrhages) are closed so
    they stay part of the retina.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = (gray > threshold).astype(np.uint8)
    k = max(3, img.shape[0] // 50) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def ben_graham_masked(
    img: np.ndarray, sigma_frac: float = 1 / 30, erode_frac: float = 0.02
) -> np.ndarray:
    """Ben Graham normalisation without edge artefacts.

    Plain Ben Graham blurs the black background into the retina's edge,
    which leaves bright and dark bands wherever the retina meets black:
    around the circle and, worse, along the flat top/bottom of photos that
    clip the retina. Those bands depend on the camera, not the disease, so
    a model can learn them as a shortcut.

    Here the local average is a normalised convolution: blur(img * mask) /
    blur(mask), i.e. the average of nearby *retina* pixels only, so the
    background never leaks in. A thin rim is masked off afterwards to drop
    anti-aliased edge pixels.
    """
    mask = fundus_mask(img)
    if mask.mean() < 0.05:  # no usable retina found; fall back to plain version
        return ben_graham(img, sigma_frac)
    # Drop the anti-aliased edge pixels (half retina, half black) first, so
    # they don't drag the local average down near the edge.
    k = max(3, int(img.shape[0] * erode_frac)) | 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    # Treat outside the image as background, so the mask also shrinks where
    # the retina touches the image border (OpenCV's default doesn't).
    inner = cv2.erode(mask, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
    sigma = img.shape[0] * sigma_frac
    m = inner.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[..., None]
    local_mean = num / np.maximum(den, 1e-3)
    out = np.clip(4 * img.astype(np.float32) - 4 * local_mean + 128, 0, 255).astype(np.uint8)
    out[inner == 0] = BEN_BACKGROUND
    return out


def crop_and_resize(img: np.ndarray, size: int) -> np.ndarray:
    """Crop the border, pad to square and resize. No colour normalisation."""
    img = pad_to_square(crop_to_fundus(img))
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def preprocess_fundus(
    img: np.ndarray,
    size: int = 512,
    use_ben_graham: bool = True,
    mask_scale: float = 0.95,
) -> np.ndarray:
    """Full preprocessing used for both training and inference."""
    img = crop_and_resize(img, size)
    if use_ben_graham:
        img = ben_graham_masked(img)
        img = circular_mask(img, scale=mask_scale, fill=BEN_BACKGROUND)
    return img


def preprocess_from_config(img: np.ndarray, pre_cfg: dict) -> np.ndarray:
    """Apply preprocessing using the ``preprocess`` section of a config."""
    return preprocess_fundus(
        img,
        size=int(pre_cfg.get("size", 512)),
        use_ben_graham=bool(pre_cfg.get("ben_graham", True)),
        mask_scale=float(pre_cfg.get("mask_scale", 0.95)),
    )


def background_value(pre_cfg: dict) -> int:
    """Pixel value to use for padding during augmentation."""
    return BEN_BACKGROUND if pre_cfg.get("ben_graham", True) else 0
