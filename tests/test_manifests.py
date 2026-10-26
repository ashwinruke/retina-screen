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
