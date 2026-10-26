"""Training loop, prediction and checkpointing."""
from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import save_config
from .data import make_loader, select
from .metrics import DEFAULT_THRESHOLDS, optimize_thresholds, qwk, scores_to_grades
from .model import RetinaModel, build_model, load_weights
from .preprocess import background_value

log = logging.getLogger(__name__)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unwrap(model: nn.Module) -> RetinaModel:
    return model.module if isinstance(model, nn.DataParallel) else model


@torch.no_grad()
def predict(
    model: nn.Module,
    loader,
    device: torch.device,
    tta: bool = False,
    amp: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (severity scores on 0..4 scale, true grades)."""
    model.eval()
    base = unwrap(model)
    use_amp = amp and device.type == "cuda"
    scores, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=use_amp):
            s = base.to_score(model(x))
            if tta:
                s = (s + base.to_score(model(torch.flip(x, dims=[3])))) / 2
        scores.append(s.float().cpu())
        targets.append(y)
    return torch.cat(scores).numpy(), torch.cat(targets).numpy()


def _loss_fn(model: RetinaModel, cfg: dict):
    if model.head_type == "regression":
        crit = nn.SmoothL1Loss()
        return lambda out, y: crit(out.float(), y.float())
    crit = nn.CrossEntropyLoss(label_smoothing=cfg["train"]["label_smoothing"])
    return lambda out, y: crit(out.float(), y)


def _lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return f


def _set_backbone_trainable(model: RetinaModel, trainable: bool) -> None:
    for p in model.backbone.parameters():
        p.requires_grad = trainable


def save_checkpoint(path: Path, model: nn.Module, cfg: dict, **extra) -> None:
    torch.save({"model": unwrap(model).state_dict(), "config": cfg, **extra}, path)


def train(cfg: dict, resume: str | None = None) -> Path:
    """Train according to ``cfg``. Returns the path of the best checkpoint."""
    tcfg = cfg["train"]
    seed_everything(tcfg["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(cfg["experiment"]["out_dir"]) / cfg["experiment"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / "config.yaml")

    # ---- data
    manifest = pd.read_csv(cfg["data"]["manifest"])
    train_df = select(manifest, cfg["data"]["train_splits"], cfg["data"]["datasets"])
    val_df = select(manifest, cfg["data"]["val_splits"], cfg["data"]["datasets"])
    if train_df.empty or val_df.empty:
        raise RuntimeError("Empty train or val set; check data.manifest / splits / datasets.")
    bg = background_value(cfg["preprocess"])
    loader_kw = dict(
        image_size=tcfg["image_size"],
        batch_size=tcfg["batch_size"],
        num_workers=tcfg["num_workers"],
        background=bg,
    )
    train_loader = make_loader(train_df, train=True, sampler_power=tcfg["sampler_power"], **loader_kw)
    val_loader = make_loader(val_df, train=False, **loader_kw)
    log.info("Train: %d images | Val: %d images | device: %s", len(train_df), len(val_df), device)
    log.info("Train grade counts: %s", np.bincount(train_df["grade"], minlength=5).tolist())

    # ---- model
    model = build_model(cfg["model"])
    if tcfg["init_from"] and not resume:
        load_weights(model, tcfg["init_from"])
    model.to(device)

    head_params = model.head_parameters()
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": tcfg["lr"] * tcfg["backbone_lr_mult"]},
            {"params": head_params, "lr": tcfg["lr"]},
        ],
        weight_decay=tcfg["weight_decay"],
    )
    steps_per_epoch = len(train_loader)
    if tcfg["max_steps_per_epoch"]:
        steps_per_epoch = min(steps_per_epoch, int(tcfg["max_steps_per_epoch"]))
    total_steps = steps_per_epoch * tcfg["epochs"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, _lr_lambda(steps_per_epoch * tcfg["warmup_epochs"], total_steps)
    )
    use_amp = tcfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    loss_fn = _loss_fn(model, cfg)

    start_epoch, best_qwk, history = 0, -1.0, []
    if resume:
        state = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        start_epoch = state["epoch"] + 1
        best_qwk = state.get("best_qwk", -1.0)
        history = state.get("history", [])
        log.info("Resumed from %s at epoch %d", resume, start_epoch)

    if tcfg["data_parallel"] and torch.cuda.device_count() > 1:
        log.info("Using DataParallel over %d GPUs", torch.cuda.device_count())
        model = nn.DataParallel(model)
    base = unwrap(model)

    # ---- loop
    for epoch in range(start_epoch, tcfg["epochs"]):
        frozen = epoch < tcfg["freeze_epochs"]
        _set_backbone_trainable(base, not frozen)
        model.train()
        t0, running, seen = time.time(), 0.0, 0
        for step, (x, y) in enumerate(train_loader):
            if step >= steps_per_epoch:
                break
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
            loss = loss_fn(out, y)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if tcfg["grad_clip"]:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item() * len(x)
            seen += len(x)
            if step % 100 == 0:
                log.info("epoch %d step %d/%d loss %.4f", epoch, step, steps_per_epoch, loss.item())

        scores, targets = predict(model, val_loader, device, tta=tcfg["tta"], amp=tcfg["amp"])
        val_qwk = qwk(targets, scores_to_grades(scores, DEFAULT_THRESHOLDS))
        thresholds = optimize_thresholds(scores, targets)
        val_qwk_tuned = qwk(targets, scores_to_grades(scores, thresholds))
        row = {
            "epoch": epoch,
            "train_loss": running / max(1, seen),
            "val_qwk": val_qwk,
            "val_qwk_tuned": val_qwk_tuned,
            "backbone_frozen": frozen,
            "seconds": round(time.time() - t0, 1),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        log.info("epoch %d | %s", epoch, json.dumps(row))

        # Select on QWK with tuned thresholds, since tuned thresholds are what
        # gets deployed. Early in training the regression output hasn't settled
        # on the 0..4 scale yet, so fixed thresholds can score a good model as 0.
        # Fitting 4 cut points on hundreds of val images adds little optimism.
        is_best = val_qwk_tuned > best_qwk
        if is_best:
            best_qwk = val_qwk_tuned
            save_checkpoint(
                out_dir / "best.pt", model, cfg,
                epoch=epoch, val_qwk=val_qwk, thresholds=thresholds,
            )
        save_checkpoint(
            out_dir / "last.pt", model, cfg,
            epoch=epoch, val_qwk=val_qwk, thresholds=thresholds, best_qwk=best_qwk,
            history=history, optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(), scaler=scaler.state_dict(),
        )
    log.info("Best val QWK %.4f -> %s", best_qwk, out_dir / "best.pt")
    return out_dir / "best.pt"


def load_for_inference(path: str | Path, device: torch.device | str = "cpu"):
    """Load a checkpoint into a ready-to-use model.

    Returns (model, config, thresholds).
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], pretrained=False)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    return model, cfg, ckpt.get("thresholds", DEFAULT_THRESHOLDS)
