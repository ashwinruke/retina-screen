"""timm backbone + GeM pooling + regression or classification head."""
from __future__ import annotations

import logging
from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


class GeM(nn.Module):
    """Generalised mean pooling. p=1 is average pooling; larger p moves toward
    max pooling, which helps when the evidence (small lesions) is sparse."""

    def __init__(self, p: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.tensor(float(p)))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.p.clamp(min=1.0)
        x = x.float().clamp(min=self.eps).pow(p)
        return F.adaptive_avg_pool2d(x, 1).pow(1.0 / p)


class RetinaModel(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        head: str = "regression",
        num_classes: int = 5,
        pretrained: bool = True,
        dropout: float = 0.3,
        drop_path: float = 0.2,
    ):
        super().__init__()
        if head not in {"regression", "classification"}:
            raise ValueError(f"Unknown head: {head}")
        self.head_type = head
        self.num_classes = num_classes
        kwargs = {"drop_path_rate": drop_path} if drop_path else {}
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, num_classes=0, global_pool="", **kwargs
        )
        self.pool = GeM()
        self.dropout = nn.Dropout(dropout)
        out_dim = 1 if head == "regression" else num_classes
        self.fc = nn.Linear(self.backbone.num_features, out_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Unpooled feature map, shape (B, C, H, W). Used by Grad-CAM."""
        return self.backbone.forward_features(x)

    def forward_head(self, fmap: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(fmap).flatten(1)
        out = self.fc(self.dropout(pooled))
        return out.squeeze(1) if self.head_type == "regression" else out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))

    def to_score(self, out: torch.Tensor) -> torch.Tensor:
        """Continuous severity score on the 0..4 scale for either head type.

        For the classification head this is the expected grade under the
        softmax, so both heads can share thresholds and metrics.
        """
        if self.head_type == "regression":
            return out.float()
        probs = out.float().softmax(dim=1)
        grades = torch.arange(self.num_classes, device=out.device, dtype=probs.dtype)
        return (probs * grades).sum(dim=1)

    def head_parameters(self):
        return list(self.pool.parameters()) + list(self.fc.parameters())


def build_model(model_cfg: dict, pretrained: bool | None = None) -> RetinaModel:
    return RetinaModel(
        backbone=model_cfg["backbone"],
        head=model_cfg["head"],
        num_classes=model_cfg.get("num_classes", 5),
        pretrained=model_cfg.get("pretrained", True) if pretrained is None else pretrained,
        dropout=model_cfg.get("dropout", 0.3),
        drop_path=model_cfg.get("drop_path", 0.2),
    )


def load_weights(model: nn.Module, path: str | Path) -> None:
    """Load matching weights from a checkpoint, skipping shape mismatches.

    Lets you initialise stage 2 from stage 1 even if the head type changed.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt.get("model", ckpt)
    state = {k.removeprefix("module."): v for k, v in state.items()}
    own = model.state_dict()
    usable = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    skipped = sorted(set(state) - set(usable))
    model.load_state_dict(usable, strict=False)
    log.info("Loaded %d tensors from %s (skipped %d).", len(usable), path, len(skipped))
    if skipped:
        log.info("Skipped: %s", skipped[:10])
