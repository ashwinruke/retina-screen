"""Preprocess every image in a manifest once and save it to disk.

Cropping, resizing and Ben Graham normalisation of multi-megapixel fundus
photos is slow; doing it once (instead of every epoch) keeps the GPU busy.
Re-running skips images that are already cached, so an interrupted run can be
resumed.

    python scripts/cache_images.py --manifest data/manifests/aptos.csv \
        --out-dir /kaggle/working/cache --config configs/finetune_aptos.yaml

Writes <out-dir>/<manifest name>.csv (paths pointing at cached images),
<out-dir>/<manifest name>.preprocess.json (the settings used) and
<out-dir>/<manifest name>_preview.jpg (a grid to eyeball).
"""
import argparse
import json
import logging
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import _bootstrap  # noqa: F401
import cv2
import numpy as np
import pandas as pd

from retina.config import load_config
from retina.preprocess import preprocess_from_config, read_rgb

log = logging.getLogger("cache_images")


def _work(item, out_root: Path, pre_cfg: dict, quality: int):
    image_id, src, dataset = item
    dst = out_root / dataset / f"{image_id}.jpg"
    if dst.exists():
        return image_id, str(dst), None
    try:
        img = preprocess_from_config(read_rgb(src), pre_cfg)
        dst.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst), cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        return image_id, str(dst), None
    except Exception as e:  # corrupt / unreadable file
        return image_id, None, f"{type(e).__name__}: {e}"


def preview_grid(df: pd.DataFrame, path: Path, n: int = 16, tile: int = 192) -> None:
    """Save a 4x4 grid of cached images, labelled with their grade."""
    per_grade = max(1, n // 5)
    sample = pd.concat([g.head(per_grade) for _, g in df.groupby("grade")]).head(n)
    tiles = []
    for _, row in sample.iterrows():
        img = cv2.resize(cv2.imread(row["image_path"]), (tile, tile))
        cv2.putText(img, f"grade {row['grade']}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        tiles.append(img)
    while len(tiles) % 4:
        tiles.append(np.zeros((tile, tile, 3), np.uint8))
    rows = [np.hstack(tiles[i : i + 4]) for i in range(0, len(tiles), 4)]
    cv2.imwrite(str(path), np.vstack(rows))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--config", help="Config whose `preprocess` section to use")
    ap.add_argument("--size", type=int, help="Override preprocess.size")
    ap.add_argument("--no-ben-graham", action="store_true", help="Disable Ben Graham normalisation")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality")
    args = ap.parse_args()

    pre_cfg = load_config(args.config)["preprocess"]
    if args.size:
        pre_cfg["size"] = args.size
    if args.no_ben_graham:
        pre_cfg["ben_graham"] = False

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_root = out_dir / "images"
    out_root.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(manifest)
    log.info("Caching %d images with %s", len(df), pre_cfg)

    items = list(zip(df["image_id"].astype(str), df["image_path"], df["dataset"]))
    work = partial(_work, out_root=out_root, pre_cfg=pre_cfg, quality=args.quality)
    results = {}
    errors = []
    with Pool(args.workers) as pool:
        for i, (image_id, dst, err) in enumerate(pool.imap_unordered(work, items, chunksize=16)):
            results[image_id] = dst
            if err:
                errors.append((image_id, err))
            if (i + 1) % 2000 == 0:
                log.info("%d / %d", i + 1, len(items))

    df["source_path"] = df["image_path"]
    df["image_path"] = df["image_id"].astype(str).map(results)
    if errors:
        log.warning("%d images failed and were dropped, e.g. %s", len(errors), errors[:3])
        df = df.dropna(subset=["image_path"])

    out_manifest = out_dir / manifest.name
    df.to_csv(out_manifest, index=False)
    with open(out_dir / f"{manifest.stem}.preprocess.json", "w") as f:
        json.dump(pre_cfg, f, indent=2)
    preview_grid(df, out_dir / f"{manifest.stem}_preview.jpg")
    log.info("Done: %s (%d images)", out_manifest, len(df))


if __name__ == "__main__":
    main()
