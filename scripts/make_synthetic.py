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
    # Optic disc at a random spot, with vessels branching out of it.
    da = rng.uniform(0, 2 * np.pi)
    disc = (int(cx + r * 0.6 * np.cos(da)), int(cy + r * 0.6 * np.sin(da)))
    thick = max(2, h // 120)
    for _ in range(6):  # vessels
        angle = rng.uniform(0, 2 * np.pi)
        end = (int(cx + r * 0.9 * np.cos(angle)), int(cy + r * 0.9 * np.sin(angle)))
        cv2.line(img, disc, end, (20, 30, 120), thick)
    cv2.circle(img, disc, max(4, r // 7), (150, 220, 250), -1)
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
        cv2.imwrite(str(root / "train_images" / f"{image_id}.png"), fake_fundus(grade, rng, 600, 800))
        rows.append((image_id, grade))
    # Duplicates like the real APTOS has: resized, re-encoded copies, one with
    # a conflicting label (both copies should be dropped) and one with the
    # same label (one copy should be kept).
    def copy_of(src_id: str, dup_id: str) -> None:
        img = cv2.imread(str(root / "train_images" / f"{src_id}.png"))
        img = cv2.resize(img, (img.shape[1] * 3 // 4, img.shape[0] * 3 // 4))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        cv2.imwrite(str(root / "train_images" / f"{dup_id}.png"), cv2.imdecode(buf, cv2.IMREAD_COLOR))

    copy_of(rows[0][0], "dup0000000000")
    rows.append(("dup0000000000", (rows[0][1] + 2) % 5))
    copy_of(rows[1][0], "dup0000000001")
    rows.append(("dup0000000001", rows[1][1]))
    pd.DataFrame(rows, columns=["id_code", "diagnosis"]).to_csv(root / "train.csv", index=False)

    # Unlabelled competition test set, plus the submission template whose
    # `diagnosis` column is all zeros. Neither must end up in the manifest.
    (root / "test_images").mkdir(exist_ok=True)
    test_ids = [f"test{i:08x}" for i in range(5)]
    for t in test_ids:
        cv2.imwrite(str(root / "test_images" / f"{t}.png"), fake_fundus(int(rng.integers(0, 5)), rng, 600, 800))
    pd.DataFrame({"id_code": test_ids}).to_csv(root / "test.csv", index=False)
    pd.DataFrame({"id_code": test_ids, "diagnosis": 0}).to_csv(root / "sample_submission.csv", index=False)


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
        # Like the real IDRiD, numbering restarts at 001 in the test set, so
        # the same file names exist in both folders.
        for i in range(n):
            grade = int(rng.integers(0, 5))
            name = f"IDRiD_{i + 1:03d}"
            cv2.imwrite(str(img_dir / f"{name}.jpg"), fake_fundus(grade, rng))
            rows.append((name, grade, int(rng.integers(0, 3)), None))
        # The Localization part reuses the same names; it must be ignored.
        loc_dir = root / "C. Localization" / "1. Original Images" / f"{letter}. {split} Set"
        loc_dir.mkdir(parents=True, exist_ok=True)
        for i in range(min(n, 3)):
            cv2.imwrite(str(loc_dir / f"IDRiD_{i + 1:03d}.jpg"), fake_fundus(0, rng))
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
