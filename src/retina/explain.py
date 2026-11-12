"""Check Grad-CAM heatmaps against lesions outlined by clinicians.

IDRiD's segmentation part has 81 photos with pixel masks for four lesion
types (microaneurysms, haemorrhages, hard and soft exudates) and the optic
disc. For each photo we ask:

* Pointing game: does the heatmap's hottest point fall on a lesion (within a
  small tolerance)? Compared with a random point on the retina.
* Energy: what share of the heatmap lies on lesions, compared with the share
  of the retina they cover? A ratio above 1 means the model looks at lesions
  more than chance.
* The same energy share for the optic disc, a known shortcut: a model can
  learn disc appearance instead of lesions.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

LESIONS = {"MA": "Microaneurysms", "HE": "Haemorrhages", "EX": "Hard exudates", "SE": "Soft exudates"}
DISC = "OD"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def _split_of(path: Path) -> str:
    return "test" if any("test" in part.lower() for part in path.parts) else "train"


def find_segmentation_set(root: str | Path) -> list[dict]:
    """Photos with their mask files, e.g. IDRiD_01.jpg + IDRiD_01_MA.tif ...

    Masks are named <image>_<TYPE>.tif. Photos and masks are paired by name
    within the same split (training/testing folder).
    """
    root = Path(root)
    masks: dict[tuple[str, str], dict[str, Path]] = {}
    photos: dict[tuple[str, str], Path] = {}
    for p in root.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        split = _split_of(p.relative_to(root))
        stem = p.stem
        kind = stem.rsplit("_", 1)[-1].upper()
        if kind in LESIONS or kind == DISC:
            masks.setdefault((split, stem.rsplit("_", 1)[0]), {})[kind] = p
        elif p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            photos[(split, stem)] = p
    items = []
    for key, kinds in sorted(masks.items()):
        if key in photos and any(k in LESIONS for k in kinds):
            items.append({"split": key[0], "name": key[1], "image_path": str(photos[key]), "masks": kinds})
    return items


def locate_segmentation_root(search_root: str | Path) -> Path:
    """Folder holding IDRiD's segmentation part, found via a *_MA.tif mask."""
    for mask in Path(search_root).rglob("*_MA.tif"):
        for parent in mask.parents:
            name = parent.name.lower()
            if "segmentation" in name and "groundtruth" not in name:
                return parent
        return mask.parents[3] if len(mask.parents) > 3 else mask.parent
    raise FileNotFoundError(f"No IDRiD segmentation masks (*_MA.tif) found under {search_root}.")


def read_mask(path: str | Path) -> np.ndarray:
    """Binary mask from an IDRiD .tif (lesions may be stored in any channel)."""
    m = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(path)
    if m.ndim == 3:
        m = m.max(axis=2)
    return m > 0


def _dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def score_heatmap(cam: np.ndarray, target: np.ndarray, retina: np.ndarray, tolerance: int) -> dict:
    """Pointing game and energy share of one heatmap against one target mask.

    Everything is restricted to the retina, so the black border can't help or
    hurt. ``chance_hit`` is the probability that a random retina pixel lands
    on the (dilated) target: the score a model that looks at random would get.
    """
    cam = np.where(retina, cam, 0)
    target = target & retina
    near = _dilate(target, tolerance) & retina
    peak = np.unravel_index(int(np.argmax(cam)), cam.shape)
    total = float(cam.sum())
    return {
        "hit": bool(near[peak]),
        "chance_hit": float(near.sum() / max(retina.sum(), 1)),
        "energy": float(cam[target].sum() / total) if total > 0 else 0.0,
        "area": float(target.sum() / max(retina.sum(), 1)),
        "peak": (int(peak[0]), int(peak[1])),
    }


def summarize(rows: list[dict]) -> dict:
    """Averages over photos, with the chance level next to each score."""
    def mean(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    out = {
        "n_images": len(rows),
        "pointing_hit_rate": mean("lesion_hit"),
        "pointing_chance": mean("lesion_chance_hit"),
        "lesion_energy": mean("lesion_energy"),
        "lesion_area": mean("lesion_area"),
        "disc_energy": mean("disc_energy"),
        "disc_area": mean("disc_area"),
        "by_type": {},
    }
    for kind, label in LESIONS.items():
        sub = [r for r in rows if r.get(f"{kind}_hit") is not None]
        if sub:
            out["by_type"][label] = {
                "n_images": len(sub),
                "hit_rate": float(np.mean([r[f"{kind}_hit"] for r in sub])),
                "chance": float(np.mean([r[f"{kind}_chance_hit"] for r in sub])),
            }
    if out["lesion_energy"] is not None and out["lesion_area"]:
        out["lesion_energy_lift"] = out["lesion_energy"] / out["lesion_area"]
    if out["disc_energy"] is not None and out["disc_area"]:
        out["disc_energy_lift"] = out["disc_energy"] / out["disc_area"]
    return out
