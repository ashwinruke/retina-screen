"""Evaluate a checkpoint, optionally refitting thresholds on a validation set,
and optionally export a slim model file for the demo.

Typical use after stage 2:

    # 1. Fit thresholds on APTOS val, report on APTOS test, export for the demo
    python scripts/evaluate.py --checkpoint outputs/finetune_aptos/best.pt \
        --manifest cache/aptos.csv --splits test \
        --fit-manifest cache/aptos.csv --fit-splits val \
        --name aptos_test --export outputs/model.pt

    # 2. External test on all of IDRiD with the thresholds frozen in model.pt
    python scripts/evaluate.py --checkpoint outputs/model.pt \
        --manifest cache/idrid.csv --splits all --name idrid_external
"""
import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401
import pandas as pd
import torch

from retina.data import make_loader, select
from retina.engine import load_for_inference, predict
from retina.metrics import DEFAULT_THRESHOLDS, full_report, optimize_thresholds, scores_to_grades
from retina.preprocess import background_value

log = logging.getLogger("evaluate")


def _scores(model, cfg, df, args, device):
    loader = make_loader(
        df,
        image_size=cfg["train"]["image_size"],
        batch_size=args.batch_size,
        train=False,
        num_workers=args.num_workers,
        background=background_value(cfg["preprocess"]),
    )
    return predict(model, loader, device, tta=args.tta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--manifest", required=True, help="Cached manifest to evaluate on")
    ap.add_argument("--splits", nargs="+", default=["test"], help='Splits to evaluate, or "all"')
    ap.add_argument("--datasets", nargs="*", help="Filter on the dataset column")
    ap.add_argument("--fit-manifest", help="Refit thresholds on this manifest (validation data only!)")
    ap.add_argument("--fit-splits", nargs="+", default=["val"])
    ap.add_argument("--no-tta", dest="tta", action="store_false")
    ap.add_argument("--name", default="eval")
    ap.add_argument("--out-dir", default="outputs/eval")
    ap.add_argument("--export", help="Write a slim model file (weights + config + thresholds) here")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, thresholds = load_for_inference(args.checkpoint, device)
    thresholds = list(thresholds)

    if args.fit_manifest:
        fit_df = select(pd.read_csv(args.fit_manifest), args.fit_splits, args.datasets)
        fit_scores, fit_y = _scores(model, cfg, fit_df, args, device)
        thresholds = optimize_thresholds(fit_scores, fit_y)
        log.info("Thresholds refit on %d images: %s", len(fit_df), [round(t, 3) for t in thresholds])

    df = select(pd.read_csv(args.manifest), args.splits, args.datasets)
    if df.empty:
        raise SystemExit("No images match --splits/--datasets in the manifest.")
    scores, y = _scores(model, cfg, df, args, device)

    report = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "splits": args.splits,
        "tta": args.tta,
        "tuned_thresholds": full_report(scores, y, thresholds),
        "default_thresholds": full_report(scores, y, DEFAULT_THRESHOLDS),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.name}.json").write_text(json.dumps(report, indent=2))
    preds = df[["image_id", "dataset", "split", "grade"]].copy()
    preds["score"] = scores
    preds["pred_grade"] = scores_to_grades(scores, thresholds)
    preds.to_csv(out / f"{args.name}_predictions.csv", index=False)

    t = report["tuned_thresholds"]
    ref = t["referable"]
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"  # noqa: E731
    print(
        f"\n{args.name}: n={t['n']}  QWK={t['qwk']:.4f}  acc={t['accuracy']:.3f}  "
        f"referable AUC={fmt(ref['auc'])} sens={fmt(ref['sensitivity'])} spec={fmt(ref['specificity'])}\n"
        f"(default thresholds QWK={report['default_thresholds']['qwk']:.4f})\n"
        f"Report: {out / (args.name + '.json')}"
    )

    if args.export:
        torch.save(
            {"model": model.state_dict(), "config": cfg, "thresholds": thresholds},
            args.export,
        )
        log.info("Exported %s", args.export)


if __name__ == "__main__":
    main()
