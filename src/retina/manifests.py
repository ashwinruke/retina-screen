"""Build one CSV "manifest" per dataset with a common schema.

Columns: image_id, image_path, grade, dataset, patient_id, split

The builders search the dataset folder for the label CSV(s) and image files
instead of relying on fixed paths, because Kaggle copies of these datasets use
different folder names.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split

log = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
COLUMNS = ["image_id", "image_path", "grade", "dataset", "patient_id", "split"]


# ---------------------------------------------------------------- helpers ---

KAGGLE_INPUT = Path("/kaggle/input")

# Columns that identify each dataset's label CSV.
LABEL_COLUMNS = {
    "aptos": {"id_code", "diagnosis"},
    "eyepacs": {"image", "level"},
    "idrid": {"Image name", "Retinopathy grade"},
}


def _listing(root: Path, depth: int = 2) -> str:
    """Folders under ``root`` up to ``depth`` levels, for error messages."""
    if not root.exists():
        return f"  ({root} does not exist)"
    lines = []
    for p in sorted(root.rglob("*")):
        rel = p.relative_to(root)
        if p.is_dir() and len(rel.parts) <= depth:
            lines.append(f"  {p}")
    return "\n".join(lines[:40]) or f"  ({root} is empty)"


def _as_paths(roots: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(roots, (str, Path)):
        roots = [roots]
    paths = [Path(r) for r in roots]
    for p in paths:
        if not p.exists():
            hint = ""
            if KAGGLE_INPUT.exists():
                hint = (
                    f"\nFolders under {KAGGLE_INPUT}:\n{_listing(KAGGLE_INPUT)}\n"
                    "Pass one of these, or pass `auto` to find it automatically."
                )
            raise FileNotFoundError(f"Dataset folder not found: {p}{hint}")
    return paths


def _has_images(folder: Path) -> bool:
    return any(p.suffix.lower() in IMAGE_EXTS for p in folder.rglob("*"))


def locate_dataset(name: str, search_root: str | Path = KAGGLE_INPUT) -> Path:
    """Find a dataset's folder by looking for its label CSV under ``search_root``.

    Starts at the folder holding the CSV and moves up until it reaches a
    folder that also contains images (IDRiD keeps labels and images in
    sibling folders). Raises if no match is found.
    """
    search_root = Path(search_root)
    required = LABEL_COLUMNS[name]
    csvs = find_label_csvs([search_root], required) if search_root.exists() else []
    if not csvs:
        raise FileNotFoundError(
            f"Could not find a {name} label CSV (columns {sorted(required)}) under "
            f"{search_root}. Is the dataset attached?\n{_listing(search_root)}"
        )
    folder = csvs[0].parent
    while not _has_images(folder) and folder != search_root and folder.parent != folder:
        folder = folder.parent
    log.info("Located %s at %s", name, folder)
    return folder


def index_images(roots: Iterable[Path]) -> dict[str, Path]:
    """Map file stem -> path for every image under the roots."""
    index: dict[str, Path] = {}
    dupes = 0
    for root in roots:
        for p in root.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                if p.stem in index:
                    dupes += 1
                    continue
                index[p.stem] = p
    if dupes:
        log.warning("Found %d images with repeated file names; kept the first of each.", dupes)
    return index


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns=lambda c: str(c).strip())
    return df.loc[:, ~df.columns.str.startswith("Unnamed")]


def find_label_csvs(roots: Iterable[Path], required: set[str]) -> list[Path]:
    """All CSVs under the roots whose header contains the required columns."""
    found = []
    for root in roots:
        for p in sorted(root.rglob("*.csv")):
            try:
                cols = set(_clean_columns(pd.read_csv(p, nrows=0)).columns)
            except Exception:  # unreadable / not a real CSV
                continue
            if required <= cols:
                found.append(p)
    return found


def _attach_paths(df: pd.DataFrame, index: dict[str, Path], name: str) -> pd.DataFrame:
    df = df.copy()
    df["image_path"] = df["image_id"].map(lambda s: str(index[s]) if s in index else None)
    missing = df["image_path"].isna().sum()
    if missing:
        log.warning("%s: %d labelled images not found on disk; dropping them.", name, missing)
        df = df.dropna(subset=["image_path"])
    if df.empty:
        raise RuntimeError(f"{name}: no labelled images found. Check the dataset path.")
    return df


def dhash(path: str | Path, size: int = 8) -> int:
    """Difference hash of an image. Identical or near-identical images collide."""
    img = cv2.imread(str(path), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if img is None:
        raise FileNotFoundError(path)
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate images. Groups with conflicting grades are dropped entirely."""
    df = df.copy()
    df["_hash"] = [dhash(p) for p in df["image_path"]]
    keep = []
    n_conflict = n_dupe = 0
    for _, group in df.groupby("_hash", sort=False):
        if len(group) == 1:
            keep.append(group.index[0])
        elif group["grade"].nunique() > 1:
            n_conflict += len(group)
        else:
            keep.append(group.index[0])
            n_dupe += len(group) - 1
    log.info(
        "Dedup: removed %d exact duplicates and %d images with conflicting labels.",
        n_dupe, n_conflict,
    )
    return df.loc[sorted(keep)].drop(columns="_hash")


def _finish(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    df = df.copy()
    df["dataset"] = dataset
    df["grade"] = df["grade"].astype(int)
    bad = ~df["grade"].between(0, 4)
    if bad.any():
        log.warning("%s: dropping %d rows with grades outside 0-4.", dataset, bad.sum())
        df = df[~bad]
    extra = [c for c in df.columns if c not in COLUMNS]
    return df[COLUMNS + extra].reset_index(drop=True)


# --------------------------------------------------------------- builders ---

def build_aptos(
    roots: str | Path | Sequence[str | Path],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    dedup: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """APTOS 2019: train.csv (id_code, diagnosis) + train_images/.

    Only the labelled training set is used; it is split into
    train / val / test, stratified by grade.
    """
    roots = _as_paths(roots)
    csvs = find_label_csvs(roots, {"id_code", "diagnosis"})
    if not csvs:
        raise RuntimeError("APTOS: no CSV with columns id_code, diagnosis found.")
    df = pd.concat([_clean_columns(pd.read_csv(p)) for p in csvs], ignore_index=True)
    df = df.drop_duplicates("id_code")
    df = df.rename(columns={"id_code": "image_id", "diagnosis": "grade"})[["image_id", "grade"]]
    df["image_id"] = df["image_id"].astype(str)
    df = _attach_paths(df, index_images(roots), "APTOS")
    if dedup:
        df = deduplicate(df)
    df["patient_id"] = df["image_id"]  # APTOS has no patient ids

    idx = np.arange(len(df))
    y = df["grade"].to_numpy()
    trainval, test = train_test_split(idx, test_size=test_frac, stratify=y, random_state=seed)
    train, val = train_test_split(
        trainval,
        test_size=val_frac / (1 - test_frac),
        stratify=y[trainval],
        random_state=seed,
    )
    split = np.empty(len(df), dtype=object)
    split[train], split[val], split[test] = "train", "val", "test"
    df["split"] = split
    return _finish(df, "aptos")


def build_eyepacs(
    roots: str | Path | Sequence[str | Path],
    val_frac: float = 0.05,
    include_test_labels: bool = True,
    seed: int = 42,
) -> pd.DataFrame:
    """Kaggle EyePACS 2015: trainLabels*.csv (image, level), plus
    retinopathy_solution.csv for the released test labels if present.

    Split train/val by patient: "10_left" and "10_right" share patient "10".
    """
    roots = _as_paths(roots)
    csvs = find_label_csvs(roots, {"image", "level"})
    if not include_test_labels:
        csvs = [p for p in csvs if "solution" not in p.name.lower()]
    if not csvs:
        raise RuntimeError("EyePACS: no CSV with columns image, level found.")
    log.info("EyePACS label files: %s", [p.name for p in csvs])
    df = pd.concat(
        [_clean_columns(pd.read_csv(p))[["image", "level"]] for p in csvs], ignore_index=True
    )
    df = df.drop_duplicates("image")
    df = df.rename(columns={"image": "image_id", "level": "grade"})
    df["image_id"] = df["image_id"].astype(str)
    df = _attach_paths(df, index_images(roots), "EyePACS")
    df["patient_id"] = df["image_id"].str.rsplit("_", n=1).str[0]

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["patient_id"]))
    split = np.empty(len(df), dtype=object)
    split[train_idx], split[val_idx] = "train", "val"
    df["split"] = split
    return _finish(df, "eyepacs")


def build_idrid(roots: str | Path | Sequence[str | Path]) -> pd.DataFrame:
    """IDRiD disease-grading subset.

    Label CSVs have columns "Image name", "Retinopathy grade" and
    "Risk of macular edema". The official train/test split is taken from the
    CSV file name.
    """
    roots = _as_paths(roots)
    csvs = find_label_csvs(roots, {"Image name", "Retinopathy grade"})
    if not csvs:
        raise RuntimeError("IDRiD: no CSV with columns 'Image name', 'Retinopathy grade' found.")
    frames = []
    for p in csvs:
        df = _clean_columns(pd.read_csv(p)).dropna(subset=["Image name"])
        name = p.name.lower()
        df["split"] = "test" if "test" in name else "train"
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    rename = {"Image name": "image_id", "Retinopathy grade": "grade"}
    if "Risk of macular edema" in df.columns:
        rename["Risk of macular edema"] = "dme_risk"
    df = df.rename(columns=rename)
    keep = ["image_id", "grade", "split"] + (["dme_risk"] if "dme_risk" in df.columns else [])
    df = df[keep]
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df = _attach_paths(df, index_images(roots), "IDRiD")
    df["patient_id"] = df["image_id"]
    return _finish(df, "idrid")


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Counts per split and grade, for a quick sanity check."""
    return df.pivot_table(
        index="split", columns="grade", values="image_id", aggfunc="count", fill_value=0
    )
