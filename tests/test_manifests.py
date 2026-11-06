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


def test_clipped_camera_eyes_are_not_grouped(tmp_path):
    """Different eyes from a camera that clips the retina's top and bottom
    share straight edges; they must not be grouped. Copies still must be."""
    import sys
    from collections import Counter

    import cv2
    import numpy as np

    from conftest import ROOT
    from retina.manifests import duplicate_groups

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_synthetic import fake_fundus

    rng = np.random.default_rng(3)
    paths = []
    for i in range(30):
        p = tmp_path / f"eye{i}.png"
        cv2.imwrite(str(p), fake_fundus(int(rng.integers(0, 5)), rng, 900, 900)[150:750])
        paths.append(str(p))
    for i in range(5):  # resized, re-encoded copies of the first five
        img = cv2.resize(cv2.imread(paths[i]), (675, 450))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        p = tmp_path / f"copy{i}.png"
        cv2.imwrite(str(p), cv2.imdecode(buf, cv2.IMREAD_COLOR))
        paths.append(str(p))

    groups = duplicate_groups(paths)
    sizes = Counter(groups)
    assert sorted(sizes.values(), reverse=True)[:6] == [2, 2, 2, 2, 2, 1]
    for i in range(5):
        assert groups[i] == groups[30 + i]
