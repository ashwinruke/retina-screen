"""Export model.pt to ONNX for the in-browser demo and check it matches PyTorch.

    pip install onnx onnxscript onnxruntime
    python scripts/export_onnx.py --model outputs/model.pt --out-dir web \
        --check-images cache/aptos.csv \
        --report outputs/eval/mvp_aptos_test.json --train-data "APTOS 2019 fundus photos"

Writes web/model.onnx and web/model.json. With --report, the test metrics from
evaluate.py are stored in model.json and shown on the demo page. With --check-images, runs a few of
the manifest's original photos through both the Python pipeline
(preprocess -> model -> Grad-CAM) and the ONNX model, and fails if the
severity scores or heatmaps disagree.
"""
import argparse
import json
import logging

import _bootstrap  # noqa: F401
import numpy as np
import pandas as pd
import torch

from retina.data import build_transforms
from retina.engine import load_for_inference
from retina.export import export_onnx
from retina.gradcam import gradcam
from retina.preprocess import background_value, crop_and_resize, preprocess_from_config, read_rgb

log = logging.getLogger("export_onnx")


def python_pipeline(model, cfg, img):
    tf = build_transforms(cfg["train"]["image_size"], train=False, background=background_value(cfg["preprocess"]))
    x = tf(image=preprocess_from_config(img, cfg["preprocess"]))["image"][None]
    with torch.no_grad():
        s = (model.to_score(model(x)) + model.to_score(model(torch.flip(x, dims=[3])))) / 2
    return float(s.clamp(0, 4)), gradcam(model, x)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", default="web")
    ap.add_argument("--check-images", help="Manifest CSV; its source_path/image_path photos are used for the check")
    ap.add_argument("--n-check", type=int, default=8)
    ap.add_argument("--tolerance", type=float, default=0.05, help="Max allowed score difference")
    ap.add_argument("--report", help="evaluate.py JSON whose tuned-threshold metrics the page should show")
    ap.add_argument("--train-data", default="APTOS 2019 fundus photos", help="Shown on the page with the metrics")
    args = ap.parse_args()

    model, cfg, thresholds = load_for_inference(args.model, "cpu")
    metrics = None
    if args.report:
        rep = json.loads(open(args.report).read())["tuned_thresholds"]
        ref = rep["referable"]
        metrics = {
            "train_data": args.train_data,
            "n_test": rep["n"],
            "qwk": rep["qwk"],
            "auc": ref["auc"],
            "sensitivity": ref["sensitivity"],
            "specificity": ref["specificity"],
        }
        if [round(t, 6) for t in rep["thresholds"]] != [round(t, 6) for t in thresholds]:
            log.warning("The report's thresholds differ from the model's; are they from the same run?")
    path = export_onnx(model, cfg, thresholds, args.out_dir, metrics=metrics)
    log.info("Wrote %s (%.1f MB) and model.json", path, path.stat().st_size / 2**20)

    if not args.check_images:
        return
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    df = pd.read_csv(args.check_images)
    col = "source_path" if "source_path" in df.columns else "image_path"
    sample = df.groupby("grade", group_keys=False).head(max(1, args.n_check // 5)).head(args.n_check)
    size = cfg["preprocess"]["size"]
    worst_score = worst_cam = 0.0
    for p in sample[col]:
        img = read_rgb(p)
        py_score, py_cam = python_pipeline(model, cfg, img)
        x = crop_and_resize(img, size).astype(np.float32).transpose(2, 0, 1)[None]
        score, cam, _ = sess.run(None, {"image": x})
        d_score = abs(float(score[0]) - py_score)
        d_cam = float(np.abs(cam[0] - py_cam).mean())
        worst_score, worst_cam = max(worst_score, d_score), max(worst_cam, d_cam)
        log.info("%-60s python %.3f  onnx %.3f  |diff| %.4f  cam diff %.4f", p[-60:], py_score, float(score[0]), d_score, d_cam)
    log.info("Worst score difference %.4f, worst mean heatmap difference %.4f", worst_score, worst_cam)
    if worst_score > args.tolerance or worst_cam > 0.05:
        raise SystemExit("ONNX output does not match PyTorch closely enough; don't deploy this export.")
    print("ONNX matches PyTorch. Ready to deploy.")


if __name__ == "__main__":
    main()
