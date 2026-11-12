"""Grad-CAM gallery and a check of the heatmaps against clinicians' lesion masks.

    python scripts/explain.py --model outputs/model.pt \
        --grading-manifest data/manifests/idrid.csv --seg-root auto \
        --out-dir docs/figures --prefix a1 --report outputs/eval/explain_a1.json

Writes:
  <prefix>_gradcam_gallery.png   two IDRiD photos per grade, picked at random
  <prefix>_gradcam_lesions.png   heatmap next to the clinicians' lesion outlines
  --report JSON (+ _per_image.csv)  pointing game and energy scores vs chance
"""
import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from retina.data import build_transforms  # noqa: E402
from retina.engine import load_for_inference  # noqa: E402
from retina.explain import (  # noqa: E402
    DISC, LESIONS, find_segmentation_set, locate_segmentation_root, read_mask, score_heatmap, summarize,
)
from retina.gradcam import gradcam, overlay  # noqa: E402
from retina.metrics import GRADE_NAMES, scores_to_grades  # noqa: E402
from retina.preprocess import (  # noqa: E402
    align_mask, background_value, crop_and_resize, fundus_mask, preprocess_from_config, read_rgb,
)

log = logging.getLogger("explain")
SURFACE, INK, INK_2 = "#fcfcfb", "#0b0b0b", "#52514e"
LESION_RGB = (0, 230, 118)  # outline colour for lesions
DISC_RGB = (255, 255, 255)


class Explainer:
    def __init__(self, model_path: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.cfg, self.thresholds = load_for_inference(model_path, self.device)
        self.size = self.cfg["preprocess"]["size"]
        self.tf = build_transforms(self.cfg["train"]["image_size"], train=False,
                                   background=background_value(self.cfg["preprocess"]))

    def __call__(self, img: np.ndarray):
        """(photo in model frame, heatmap, severity score, grade)."""
        x = self.tf(image=preprocess_from_config(img, self.cfg["preprocess"]))["image"][None].to(self.device)
        with torch.no_grad():
            score = float(self.model.to_score(self.model(x)).clamp(0, 4))
        cam = gradcam(self.model, x)
        if cam.shape != (self.size, self.size):  # model input smaller than the preprocessed photo
            cam = cv2.resize(cam, (self.size, self.size), interpolation=cv2.INTER_LINEAR)
        grade = int(scores_to_grades(np.array([score]), self.thresholds)[0])
        return crop_and_resize(img, self.size), cam, score, grade


def _outline(rgb: np.ndarray, mask: np.ndarray, colour, thickness: int = 2) -> np.ndarray:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = rgb.copy()
    cv2.drawContours(out, contours, -1, colour, thickness)
    return out


def gallery(ex: Explainer, manifest: str, path: Path, per_grade: int, seed: int) -> None:
    df = pd.read_csv(manifest)
    picks = pd.concat([g.sample(min(per_grade, len(g)), random_state=seed) for _, g in df.groupby("grade")])
    fig, axes = plt.subplots(per_grade, 5, figsize=(15, 3.6 * per_grade + 0.5), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax in axes.flat:
        ax.axis("off")
    counters = {g: 0 for g in range(5)}
    for _, row in picks.iterrows():
        g = int(row["grade"])
        photo, cam, score, pred = ex(read_rgb(row["image_path"]))
        ax = axes[counters[g], g]
        counters[g] += 1
        ax.imshow(overlay(photo, cam))
        ax.set_title(f"true {g} {GRADE_NAMES[g]}\npredicted {pred} (score {score:.2f})", fontsize=9, color=INK)
    fig.suptitle("Grad-CAM on IDRiD (never seen in training): warmer areas raised the severity score",
                 x=0.01, ha="left", fontsize=12, color=INK, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96), h_pad=2.0)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def lesion_check(ex: Explainer, items: list[dict], tolerance: int):
    rows, examples = [], []
    for item in items:
        img = read_rgb(item["image_path"])
        photo, cam, score, pred = ex(img)
        retina = fundus_mask(photo) > 0
        masks = {k: align_mask(read_mask(p), img, ex.size) for k, p in item["masks"].items()}
        lesions = np.zeros_like(retina)
        for k in LESIONS:
            if k in masks:
                lesions |= masks[k]
        row = {"name": item["name"], "split": item["split"], "score": score, "pred_grade": pred}
        s = score_heatmap(cam, lesions, retina, tolerance)
        row.update({"lesion_hit": s["hit"], "lesion_chance_hit": s["chance_hit"],
                    "lesion_energy": s["energy"], "lesion_area": s["area"]})
        for k in LESIONS:
            if k in masks and masks[k].any():
                t = score_heatmap(cam, masks[k], retina, tolerance)
                row[f"{k}_hit"], row[f"{k}_chance_hit"] = t["hit"], t["chance_hit"]
        if DISC in masks and masks[DISC].any():
            d = score_heatmap(cam, masks[DISC], retina, 0)
            row["disc_energy"], row["disc_area"] = d["energy"], d["area"]
        rows.append(row)
        examples.append((item["name"], photo, cam, lesions, masks.get(DISC), s["peak"], s["hit"]))
    return rows, examples


def lesion_figure(examples, path: Path, n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    chosen = [examples[i] for i in sorted(rng.choice(len(examples), size=min(n, len(examples)), replace=False))]
    fig, axes = plt.subplots(len(chosen), 3, figsize=(11, 3.7 * len(chosen)), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for row, (name, photo, cam, lesions, disc, peak, hit) in zip(axes, chosen):
        marked = _outline(photo, lesions, LESION_RGB)
        if disc is not None:
            marked = _outline(marked, disc, DISC_RGB)
        heat = overlay(photo, cam)
        cv2.drawMarker(heat, (peak[1], peak[0]), (255, 255, 255), cv2.MARKER_CROSS, 28, 3)
        cv2.drawMarker(marked, (peak[1], peak[0]), (255, 255, 255), cv2.MARKER_CROSS, 28, 3)
        for ax, im, title in zip(row, (photo, heat, marked),
                                 (name, "Grad-CAM (+ hottest point)",
                                  f"Lesions (green), disc (white): {'hit' if hit else 'miss'}")):
            ax.imshow(im)
            ax.set_title(title, fontsize=10, color=INK)
            ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--grading-manifest", help="Manifest for the gallery (e.g. data/manifests/idrid.csv)")
    ap.add_argument("--seg-root", default="auto", help="IDRiD 'A. Segmentation' folder, or auto")
    ap.add_argument("--search-root", default="/kaggle/input", help="Where --seg-root auto looks")
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--prefix", default="a1")
    ap.add_argument("--report", default="outputs/eval/explain.json")
    ap.add_argument("--tolerance", type=int, default=16, help="Pointing-game tolerance in pixels (of 512)")
    ap.add_argument("--per-grade", type=int, default=2)
    ap.add_argument("--n-examples", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ex = Explainer(args.model)

    if args.grading_manifest:
        gallery(ex, args.grading_manifest, out / f"{args.prefix}_gradcam_gallery.png", args.per_grade, args.seed)
        log.info("Gallery written")

    root = locate_segmentation_root(args.search_root) if args.seg_root == "auto" else Path(args.seg_root)
    items = find_segmentation_set(root)
    if not items:
        raise SystemExit(f"No photo/mask pairs found under {root}")
    log.info("Segmentation set: %d photos from %s", len(items), root)
    rows, examples = lesion_check(ex, items, args.tolerance)
    lesion_figure(examples, out / f"{args.prefix}_gradcam_lesions.png", args.n_examples, args.seed)

    summary = summarize(rows)
    summary.update({"tolerance_px": args.tolerance, "model": args.model, "seg_root": str(root)})
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))
    pd.DataFrame(rows).to_csv(report.with_name(report.stem + "_per_image.csv"), index=False)

    s = summary
    print(f"\nPhotos: {s['n_images']}")
    print(f"Pointing game: hottest point on a lesion in {s['pointing_hit_rate']:.0%} of photos "
          f"(random point: {s['pointing_chance']:.0%})")
    print(f"Heatmap on lesions: {s['lesion_energy']:.1%} of heat on {s['lesion_area']:.1%} of the retina "
          f"({s.get('lesion_energy_lift', float('nan')):.1f}x chance)")
    if s["disc_energy"] is not None:
        print(f"Heatmap on optic disc: {s['disc_energy']:.1%} of heat on {s['disc_area']:.1%} of the retina "
              f"({s.get('disc_energy_lift', float('nan')):.1f}x chance)")
    for label, t in s["by_type"].items():
        print(f"  {label:15s} n={t['n_images']:3d}  hit {t['hit_rate']:.0%}  (random {t['chance']:.0%})")
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
