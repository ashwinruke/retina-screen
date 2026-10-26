"""Build manifests (CSV of image path, grade, patient, split) for each dataset.

Point each flag at the folder you attached on Kaggle (or downloaded locally).
Folder names inside don't matter: label CSVs and images are found by search.

    python scripts/prepare_data.py \
        --aptos /kaggle/input/aptos2019-blindness-detection \
        --eyepacs /kaggle/input/<eyepacs-folder> \
        --idrid /kaggle/input/<idrid-folder> \
        --out-dir data/manifests
"""
import argparse
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from retina import manifests

log = logging.getLogger("prepare_data")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aptos", nargs="+", help="APTOS 2019 folder(s)")
    ap.add_argument("--eyepacs", nargs="+", help="EyePACS 2015 folder(s); pass train and test folders if separate")
    ap.add_argument("--idrid", nargs="+", help="IDRiD folder(s) containing the disease-grading subset")
    ap.add_argument("--out-dir", default="data/manifests")
    ap.add_argument("--no-dedup", action="store_true", help="Skip APTOS duplicate removal")
    ap.add_argument("--eyepacs-train-only", action="store_true", help="Ignore released EyePACS test labels")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if not (args.aptos or args.eyepacs or args.idrid):
        ap.error("Pass at least one of --aptos, --eyepacs, --idrid")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    if args.aptos:
        jobs.append(("aptos", lambda: manifests.build_aptos(args.aptos, dedup=not args.no_dedup, seed=args.seed)))
    if args.eyepacs:
        jobs.append(("eyepacs", lambda: manifests.build_eyepacs(
            args.eyepacs, include_test_labels=not args.eyepacs_train_only, seed=args.seed)))
    if args.idrid:
        jobs.append(("idrid", lambda: manifests.build_idrid(args.idrid)))

    for name, build in jobs:
        df = build()
        path = out / f"{name}.csv"
        df.to_csv(path, index=False)
        log.info("%s: %d images -> %s\n%s", name, len(df), path, manifests.summarize(df).to_string())


if __name__ == "__main__":
    main()
