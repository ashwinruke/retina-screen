"""Export a trained model to ONNX for the in-browser demo.

The browser only crops the black border, pads to a square and resizes to
512 x 512 (see web/app.js). Everything else runs inside the ONNX graph, so it
matches the Python pipeline exactly:

  1. Ben Graham normalisation with the retina-only local average
     (``preprocess.ben_graham_masked``) and the circular mask
  2. ImageNet normalisation
  3. The network, with horizontal-flip TTA
  4. Grad-CAM, computed in closed form (no backward pass is needed, because
     the head is GeM pooling followed by one linear layer)

Inputs:  image   float32 (1, 3, S, S), RGB 0-255, already cropped/padded/resized
Outputs: score   float32 (1,)          severity on the 0-4 scale (clamped)
         cam     float32 (1, S, S)      Grad-CAM heatmap in [0, 1]
         seen    float32 (1, 3, S, S)   the normalised image the network sees, 0-255
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .metrics import GRADE_NAMES, REFERABLE_GRADE
from .model import RetinaModel
from .preprocess import BEN_BACKGROUND

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _ellipse(k: int) -> torch.Tensor:
    return torch.from_numpy(cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)).astype(np.float32))


class BrowserModel(nn.Module):
    """Preprocessing (after crop/resize) + model + TTA + Grad-CAM in one graph."""

    def __init__(self, model: RetinaModel, pre_cfg: dict):
        super().__init__()
        if model.head_type != "regression":
            raise ValueError("Browser export supports the regression head only.")
        self.model = model.eval()
        size = int(pre_cfg.get("size", 512))
        self.size = size
        self.use_ben_graham = bool(pre_cfg.get("ben_graham", True))

        # Same structuring elements and blur kernel as preprocess.py / OpenCV.
        k_close = max(3, size // 50) | 1
        k_erode = max(3, int(size * 0.02)) | 1
        self.register_buffer("k_close", _ellipse(k_close)[None, None])
        self.register_buffer("k_erode", _ellipse(k_erode)[None, None])
        sigma = size / 30
        ksize = int(round(sigma * 4 * 2 + 1)) | 1  # OpenCV's choice for float images
        g = cv2.getGaussianKernel(ksize, sigma).astype(np.float32).flatten()
        self.register_buffer("gauss", torch.from_numpy(g))

        # Circular mask, drawn with OpenCV so the edge pixels match exactly.
        circle = np.zeros((size, size), np.uint8)
        scale = float(pre_cfg.get("mask_scale", 0.95))
        cv2.circle(circle, (size // 2, size // 2), int(size / 2 * scale), 1, thickness=-1)
        self.register_buffer("circle", torch.from_numpy(circle.astype(np.float32))[None, None])

        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1) * 255)
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1) * 255)

    # ---- morphology and blur, matching OpenCV's border rules

    def _dilate(self, m: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        p = k.shape[-1] // 2
        return (F.conv2d(F.pad(m, (p, p, p, p), value=0.0), k) > 0.5).float()

    def _erode(self, m: torch.Tensor, k: torch.Tensor, border: float) -> torch.Tensor:
        p = k.shape[-1] // 2
        hits = F.conv2d(F.pad(m, (p, p, p, p), value=border), k)
        return (hits > k.sum() - 0.5).float()

    def _blur(self, x: torch.Tensor) -> torch.Tensor:
        """Separable Gaussian blur with reflect-101 borders (cv2.GaussianBlur)."""
        c = x.shape[1]
        p = self.gauss.numel() // 2
        kh = self.gauss.view(1, 1, 1, -1).repeat(c, 1, 1, 1)
        kv = self.gauss.view(1, 1, -1, 1).repeat(c, 1, 1, 1)
        x = F.conv2d(F.pad(x, (p, p, 0, 0), mode="reflect"), kh, groups=c)
        return F.conv2d(F.pad(x, (0, 0, p, p), mode="reflect"), kv, groups=c)

    def ben_graham(self, img: torch.Tensor) -> torch.Tensor:
        r, g, b = img[:, 0:1], img[:, 1:2], img[:, 2:3]
        gray = torch.floor((r * 4899 + g * 9617 + b * 1868 + 8192) / 16384)  # cv2 RGB2GRAY
        mask = (gray > 10).float()
        mask = self._erode(self._dilate(mask, self.k_close), self.k_close, border=1.0)  # MORPH_CLOSE
        inner = self._erode(mask, self.k_erode, border=0.0)
        local = self._blur(img * inner) / torch.clamp(self._blur(inner), min=1e-3)
        out = torch.floor(torch.clamp(4 * img - 4 * local + 128, 0, 255))
        keep = inner * self.circle
        return out * keep + BEN_BACKGROUND * (1 - keep)

    def forward(self, image: torch.Tensor):
        seen = self.ben_graham(image) if self.use_ben_graham else image
        x = (seen - self.mean) / self.std

        m = self.model
        fmap = m.forward_features(x).float()
        out = m.forward_head(fmap)
        out_flip = m(torch.flip(x, dims=[3]))
        score = torch.clamp((out + out_flip) / 2, 0, 4)

        # Grad-CAM for out = w . GeM(fmap) + b, written out by hand:
        #   GeM_c = (mean_ij f^p)^(1/p),  d out / d f_cij = w_c * GeM_c^(1-p) * f_cij^(p-1) / HW
        p = torch.clamp(m.pool.p, min=1.0)
        f = torch.clamp(fmap, min=m.pool.eps)
        hw = fmap.shape[2] * fmap.shape[3]
        pooled = f.pow(p).mean(dim=(2, 3), keepdim=True).pow(1.0 / p)
        active = (fmap > m.pool.eps).float()  # clamp blocks the gradient below eps
        w = m.fc.weight.view(1, -1, 1, 1)
        grads = w * pooled.pow(1 - p) * f.pow(p - 1) * active / hw
        weights = grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * fmap).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=(self.size, self.size), mode="bilinear", align_corners=False)
        cam = cam / torch.clamp(cam.amax(dim=(2, 3), keepdim=True), min=1e-12)
        return score, cam[:, 0], seen


def display_name(backbone: str) -> str:
    """'efficientnet_b3' -> 'EfficientNet-B3'; other timm names are returned as-is."""
    if backbone.startswith("efficientnet_"):
        return "EfficientNet-" + backbone.split("_", 1)[1].upper()
    return backbone


def export_onnx(
    model: RetinaModel,
    cfg: dict,
    thresholds,
    out_dir: str | Path,
    metrics: dict | None = None,
    opset: int = 18,
) -> Path:
    """Write model.onnx and model.json (thresholds, labels, input size and,
    optionally, test metrics shown on the demo page) to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wrapper = BrowserModel(model.float().cpu().eval(), cfg["preprocess"]).eval()
    size = wrapper.size
    dummy = torch.rand(1, 3, size, size) * 255
    path = out_dir / "model.onnx"
    kwargs = dict(input_names=["image"], output_names=["score", "cam", "seen"], opset_version=opset)
    with torch.no_grad():
        try:
            torch.onnx.export(wrapper, (dummy,), str(path), dynamo=True, external_data=False, **kwargs)
        except Exception as err:  # older/newer exporters differ; fall back to TorchScript
            print(f"dynamo exporter failed ({type(err).__name__}: {err}); using the legacy exporter")
            torch.onnx.export(wrapper, (dummy,), str(path), dynamo=False, **kwargs)
    meta = {
        "input_size": size,
        "thresholds": [float(t) for t in thresholds],
        "grade_names": GRADE_NAMES,
        "referable_grade": REFERABLE_GRADE,
        "backbone": cfg["model"]["backbone"],
        "model_name": display_name(cfg["model"]["backbone"]),
        "crop_threshold": 10,
    }
    if metrics:
        meta["metrics"] = metrics
    (out_dir / "model.json").write_text(json.dumps(meta, indent=2))
    return path
