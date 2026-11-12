"""Split a dataset's training images into train and val, keeping its test set.

Used for A4: IDRiD's 413 training images become ~330 train + ~83 val
(stratified by grade), so the epoch and the cut points are chosen on local
data the model didn't train on, and the 103 official test images stay unseen.

    python scripts/make_local_split.py --manifest data/manifests/idrid.csv \
        --out data/manifests/idrid_local.csv --val-frac 0.2
"""
import argparse
import logging

import _bootstrap  # noqa: F401
import pandas as pd
from sklearn.model_selection import train_test_split

log = logging.getLogger("make_local_split")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    train = df[df["split"] == "train"]
    if train.empty:
        raise SystemExit("No 'train' rows in the manifest.")
    counts = train["grade"].value_counts()
    stratify = train["grade"] if (counts >= 2).all() else None
    if stratify is None:
        log.warning("Some grades have fewer than 2 training images (%s); splitting without stratifying.",
                    counts[counts < 2].to_dict())
    _, val_idx = train_test_split(train.index, test_size=args.val_frac, stratify=stratify,
                                  random_state=args.seed)
    df.loc[val_idx, "split"] = "val"
    df.to_csv(args.out, index=False)
    log.info("%s -> %s\n%s", args.manifest, args.out,
             df.pivot_table(index="split", columns="grade", values="image_id", aggfunc="count", fill_value=0))


if __name__ == "__main__":
    main()
