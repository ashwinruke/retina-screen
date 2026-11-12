"""Lesion-mask alignment and heatmap scoring."""
import cv2
import numpy as np
import pandas as pd
import torch

from conftest import ROOT, run
from retina.explain import find_segmentation_set, locate_segmentation_root, read_mask, score_heatmap
from retina.preprocess import align_mask, crop_and_resize


def test_align_mask_lands_on_the_same_pixels():
    """A lesion drawn on a clipped, off-centre photo must line up with its
    mask after the model's crop / pad / resize."""
    img = np.zeros((700, 1100, 3), np.uint8)
    cv2.circle(img, (520, 350), 420, (40, 90, 200), -1)  # clipped top and bottom
    mask = np.zeros(img.shape[:2], np.uint8)
    for c in [(400, 300), (650, 420), (520, 150)]:
        cv2.circle(img, c, 12, (255, 255, 255), -1)
        cv2.circle(mask, c, 12, 255, -1)
    photo = crop_and_resize(img, 256)
    m = align_mask(mask, img, 256)
    bright = photo.min(axis=2) > 200
    assert m.sum() > 0
    overlap = (m & bright).sum() / bright.sum()
    assert overlap > 0.9


def test_score_heatmap():
    retina = np.zeros((64, 64), bool)
    retina[8:56, 8:56] = True
    target = np.zeros_like(retina)
    target[20:24, 30:34] = True
    cam = np.zeros((64, 64), np.float32)
    cam[21, 31] = 1.0
    s = score_heatmap(cam, target, retina, tolerance=2)
    assert s["hit"] and s["energy"] == 1.0
    assert 0 < s["chance_hit"] < 0.1
    cam2 = np.zeros_like(cam)
    cam2[50, 10] = 1.0
    assert not score_heatmap(cam2, target, retina, tolerance=2)["hit"]


def test_find_segmentation_set(synthetic):
    root = locate_segmentation_root(synthetic / "idrid")
    assert root.name == "A. Segmentation"
    items = find_segmentation_set(root)
    assert len(items) == 6
    assert {i["split"] for i in items} == {"train", "test"}
    assert set(items[0]["masks"]) == {"MA", "HE", "EX", "OD"}
    assert read_mask(items[0]["masks"]["OD"]).any()


def test_explain_script(synthetic, manifests_dir, tmp_path):
    from retina.config import load_config
    from retina.model import build_model

    cfg = load_config(ROOT / "configs/smoke.yaml")
    model = build_model(cfg["model"], pretrained=False)
    torch.save({"model": model.state_dict(), "config": cfg, "thresholds": [0.5, 1.5, 2.5, 3.5]},
               tmp_path / "model.pt")
    out = run(ROOT / "scripts/explain.py", "--model", tmp_path / "model.pt",
              "--grading-manifest", manifests_dir / "idrid.csv",
              "--seg-root", synthetic / "idrid" / "A. Segmentation",
              "--out-dir", tmp_path / "fig", "--report", tmp_path / "explain.json", "--per-grade", 1)
    assert "Pointing game" in out
    assert (tmp_path / "fig" / "a1_gradcam_gallery.png").exists()
    assert (tmp_path / "fig" / "a1_gradcam_lesions.png").exists()
    rows = pd.read_csv(tmp_path / "explain_per_image.csv")
    assert len(rows) == 6 and rows["lesion_energy"].between(0, 1).all()
