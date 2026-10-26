"""Generate tiny fake versions of APTOS, EyePACS and IDRiD with the same folder
layouts and label files as the real datasets.

Used by the tests, and handy for dry-running the whole pipeline on a laptop
before spending Kaggle GPU time. The "retinas" are orange discs with bright
spots whose number grows with the grade, so a model can actually learn
something from them.

    python scripts/make_synthetic.py --out data/synthetic
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def fake_fundus(grade: int, rng: np.random.Generator, h: int = 300, w: int = 400) -> np.ndarray:
    img = np.zeros((h, w, 3), np.uint8)
    r = int(min(h, w) * rng.uniform(0.42, 0.48))
    cx, cy = w // 2 + int(rng.integers(-10, 10)), h // 2 + int(rng.integers(-5, 5))
    base = np.array([40, 80, 200]) * rng.uniform(0.7, 1.1)  # BGR orange-red
    cv2.circle(img, (cx, cy), r, tuple(int(c) for c in base), -1)
    for _ in range(6):  # vessels
        angle = rng.uniform(0, 2 * np.pi)
        end = (int(cx + r * 0.9 * np.cos(angle)), int(cy + r * 0.9 * np.sin(angle)))
        cv2.line(img, (cx, cy), end, (20, 30, 120), 2)
    for _ in range(grade * 8):  # "lesions"
        a, d = rng.uniform(0, 2 * np.pi), rng.uniform(0, r * 0.8)
        p = (int(cx + d * np.cos(a)), int(cy + d * np.sin(a)))
        cv2.circle(img, p, int(rng.integers(2, 5)), (120, 230, 240), -1)
    noise = rng.normal(0, 4, img.shape)
    mask = img.sum(-1, keepdims=True) > 0
    return np.clip(img + noise * mask, 0, 255).astype(np.uint8)


def make_aptos(root: Path, n: int, rng) -> None:
    (root / "train_images").mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        grade = int(rng.integers(0, 5))
        image_id = f"{i:012x}"
        cv2.imwrite(str(root / "train_images" / f"{image_id}.png"), fake_fundus(grade, rng))
        rows.append((image_id, grade))
    # A duplicate with a conflicting label, like the real APTOS has.
    src = root / "train_images" / f"{rows[0][0]}.png"
    dup_id = "dup0000000000"
    cv2.imwrite(str(root / "train_images" / f"{dup_id}.png"), cv2.imread(str(src)))
    rows.append((dup_id, (rows[0][1] + 2) % 5))
    pd.DataFrame(rows, columns=["id_code", "diagnosis"]).to_csv(root / "train.csv", index=False)
    pd.DataFrame({"id_code": ["unlabelled"]}).to_csv(root / "test.csv", index=False)


def make_eyepacs(root: Path, n_patients: int, rng) -> None:
    (root / "train").mkdir(parents=True, exist_ok=True)
    rows = []
    for p in range(n_patients):
        for eye in ("left", "right"):
            grade = int(rng.choice(5, p=[0.6, 0.1, 0.15, 0.08, 0.07]))
            name = f"{p + 10}_{eye}"
            cv2.imwrite(str(root / "train" / f"{name}.jpeg"), fake_fundus(grade, rng))
            rows.append((name, grade))
    pd.DataFrame(rows, columns=["image", "level"]).to_csv(root / "trainLabels.csv", index=False)


def make_idrid(root: Path, n_train: int, n_test: int, rng) -> None:
    base = root / "B. Disease Grading"
    for split, n, letter in (("Training", n_train, "a"), ("Testing", n_test, "b")):
        img_dir = base / "1. Original Images" / f"{letter}. {split} Set"
        img_dir.mkdir(parents=True, exist_ok=True)
        rows = []
        offset = 0 if split == "Training" else n_train
        for i in range(n):
            grade = int(rng.integers(0, 5))
            name = f"IDRiD_{offset + i + 1:03d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), fake_fundus(grade, rng))
            rows.append((name, grade, int(rng.integers(0, 3)), None))
        gt = base / "2. Groundtruths"
        gt.mkdir(parents=True, exist_ok=True)
        # Real file has a trailing space in the DME column and blank columns.
        pd.DataFrame(
            rows, columns=["Image name", "Retinopathy grade", "Risk of macular edema ", ""]
        ).to_csv(gt / f"{letter}. IDRiD_Disease Grading_{split} Labels.csv", index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--aptos", type=int, default=60)
    ap.add_argument("--eyepacs-patients", type=int, default=40)
    ap.add_argument("--idrid", type=int, nargs=2, default=[20, 10], metavar=("TRAIN", "TEST"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    make_aptos(out / "aptos", args.aptos, rng)
    make_eyepacs(out / "eyepacs", args.eyepacs_patients, rng)
    make_idrid(out / "idrid", *args.idrid, rng)
    print(f"Synthetic datasets written to {out}")


if __name__ == "__main__":
    main()
