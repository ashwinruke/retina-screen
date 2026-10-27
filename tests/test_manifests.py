import pandas as pd

from retina.manifests import COLUMNS


def test_aptos_dedup_and_splits(manifests_dir):
    df = pd.read_csv(manifests_dir / "aptos.csv")
    assert list(df.columns[: len(COLUMNS)]) == COLUMNS
    # 60 images + 1 conflicting duplicate of image 0 -> both copies dropped
    assert len(df) == 59
    assert "dup0000000000" not in set(df["image_id"])
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
