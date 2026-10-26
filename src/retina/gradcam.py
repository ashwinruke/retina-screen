"""Grad-CAM on the backbone's last feature map."""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from .model import RetinaModel


def gradcam(
    model: RetinaModel, x: torch.Tensor, target_class: int | None = None
) -> np.ndarray:
    """Heatmap in [0, 1] with the same height/width as ``x``.

    ``x`` is a single preprocessed image tensor of shape (1, 3, H, W).
    For the regression head the target is the severity score itself, so the
    map shows what pushed the score up. For the classification head the
    target is ``target_class`` (default: the predicted class).

    Gradients are taken with respect to the feature map only, so no backward
    pass through the backbone is needed.
    """
    if x.ndim != 4 or x.shape[0] != 1:
        raise ValueError("gradcam expects a single image batch of shape (1, 3, H, W)")
    model.eval()
    with torch.no_grad():
        fmap = model.forward_features(x)
    fmap = fmap.detach().float().requires_grad_(True)
    with torch.enable_grad():
        out = model.forward_head(fmap)
        if model.head_type == "regression":
            target = out.sum()
        else:
            cls = int(out.argmax(1)) if target_class is None else int(target_class)
            target = out[0, cls]
        (grads,) = torch.autograd.grad(target, fmap)

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)
    cam = cam[0, 0].detach().cpu().numpy()
    peak = cam.max()
    return cam / peak if peak > 0 else cam


def overlay(image_rgb: np.ndarray, cam: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a heatmap over an RGB uint8 image."""
    if cam.shape != image_rgb.shape[:2]:
        cam = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]))
    heat = cv2.applyColorMap(np.uint8(255 * np.clip(cam, 0, 1)), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(image_rgb, 1 - alpha, heat, alpha, 0)
