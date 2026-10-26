"""Train a model from a config, with optional key=value overrides.

    python scripts/train.py --config configs/pretrain_eyepacs.yaml
    python scripts/train.py --config configs/finetune_aptos.yaml train.epochs=5
    python scripts/train.py --config configs/pretrain_eyepacs.yaml \
        --resume outputs/pretrain_eyepacs/last.pt   # continue in a new Kaggle session
"""
import argparse
import json
import logging
from pathlib import Path

import _bootstrap  # noqa: F401

from retina.config import load_config
from retina.engine import train

log = logging.getLogger("train")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", help="last.pt from an interrupted run")
    ap.add_argument("overrides", nargs="*", help="section.key=value")
    args = ap.parse_args()

    cfg = load_config(args.config, args.overrides)

    # Use the preprocessing settings the cache was actually built with, so the
    # checkpoint (and the demo) applies exactly the same preprocessing.
    sidecar = Path(cfg["data"]["manifest"]).with_suffix(".preprocess.json")
    if sidecar.exists():
        cached = json.loads(sidecar.read_text())
        if cached != cfg["preprocess"]:
            log.warning("Config preprocess %s differs from cache %s; using the cache's settings.",
                        cfg["preprocess"], cached)
        cfg["preprocess"] = cached

    train(cfg, resume=args.resume)


if __name__ == "__main__":
    main()
