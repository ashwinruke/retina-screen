import pandas as pd

from retina.manifests import COLUMNS


def test_aptos_dedup_and_splits(manifests_dir):
    df = pd.read_csv(manifests_dir / "aptos.csv")
    assert list(df.columns[: len(COLUMNS)]) == COLUMNS
    # 60 images + a conflicting copy of image 0 (both dropped) + a same-label
    # copy of image 1 (one kept). sample_submission.csv must be ignored.
    assert len(df) == 59
    ids = set(df["image_id"])
    assert "dup0000000000" not in ids
    assert len(ids & {df["image_id"].iloc[0], "dup0000000001"}) <= 1
    assert not any(i.startswith("test") for i in ids)
    assert not df["image_path"].str.contains("test_images").any()
    assert set(df["split"]) == {"train", "val", "test"}
    assert df["image_id"].is_unique


def test_eyepacs_split_by_patient(manifests_dir):
    df = pd.read_csv(manifests_dir / "eyepacs.csv", dtype={"patient_id": str})
    assert len(df) == 80
    per_patient = df.groupby("patient_id")["split"].nunique()
    assert (per_patient == 1).all(), "a patient's two eyes landed in different splits"
    assert set(df["split"]) == {"train", "val"}


def test_idrid_official_split(manifests_dir):
    df = pd.read_csv(manifests_dir / "idrid.csv")
    assert df["split"].value_counts().to_dict() == {"train": 20, "test": 10}
    assert "dme_risk" in df.columns
    assert df["grade"].between(0, 4).all()


def test_locate_dataset(synthetic):
    from retina.manifests import locate_dataset

    assert locate_dataset("aptos", synthetic) == synthetic / "aptos"
    assert locate_dataset("eyepacs", synthetic) == synthetic / "eyepacs"
    # IDRiD labels live in "2. Groundtruths"; images in a sibling folder.
    assert locate_dataset("idrid", synthetic) == synthetic / "idrid" / "B. Disease Grading"


def test_auto_flag_and_helpful_error(synthetic, tmp_path):
    from conftest import ROOT, run

    out = run(ROOT / "scripts/prepare_data.py", "--aptos", "auto", "--idrid", "auto",
              "--search-root", synthetic, "--out-dir", tmp_path)
    assert (tmp_path / "aptos.csv").exists() and (tmp_path / "idrid.csv").exists()
    assert "Located aptos" in out

    import pytest
    with pytest.raises(AssertionError, match="Could not find a eyepacs label CSV"):
        run(ROOT / "scripts/prepare_data.py", "--eyepacs", "auto",
            "--search-root", synthetic / "aptos", "--out-dir", tmp_path)


def test_fingerprint_matches_copies_not_different_eyes(synthetic):
    import numpy as np

    from retina.manifests import DUPLICATE_THRESHOLD, fingerprint

    imgs = synthetic / "aptos" / "train_images"
    train = pd.read_csv(synthetic / "aptos" / "train.csv")
    a = fingerprint(imgs / f"{train.id_code[1]}.png")
    copy = fingerprint(imgs / "dup0000000001.png")
    assert float(a @ copy) > DUPLICATE_THRESHOLD
    others = np.stack([fingerprint(imgs / f"{i}.png") for i in train.id_code[2:40]])
    assert (others @ a).max() < DUPLICATE_THRESHOLD
