"""End to end on synthetic data: prepare -> cache -> train -> evaluate -> export -> infer."""
import json

import numpy as np
import pandas as pd
import torch

from conftest import ROOT, run


def test_full_pipeline(synthetic, manifests_dir, tmp_path):
    cache = tmp_path / "cache"
    for name in ("aptos", "idrid"):
        run(
            ROOT / "scripts/cache_images.py",
            "--manifest", manifests_dir / f"{name}.csv",
            "--out-dir", cache,
            "--config", ROOT / "configs/smoke.yaml",
            "--workers", 2,
        )
    cached = pd.read_csv(cache / "aptos.csv")
    assert all(p.endswith(".jpg") for p in cached["image_path"])
    assert (cache / "aptos_preview.jpg").exists()
    assert json.loads((cache / "aptos.preprocess.json").read_text())["size"] == 128

    out = tmp_path / "outputs"
    log = run(
        ROOT / "scripts/train.py", "--config", ROOT / "configs/smoke.yaml",
        f"data.manifest={cache / 'aptos.csv'}", f"experiment.out_dir={out}",
    )
    assert "Best val QWK" in log
    best = out / "smoke" / "best.pt"
    assert best.exists() and (out / "smoke" / "history.csv").exists()

    # Resume continues from the saved epoch instead of starting over.
    log = run(
        ROOT / "scripts/train.py", "--config", ROOT / "configs/smoke.yaml",
        f"data.manifest={cache / 'aptos.csv'}", f"experiment.out_dir={out}",
        "train.epochs=3", "--resume", out / "smoke" / "last.pt",
    )
    assert "Resumed" in log
    assert len(pd.read_csv(out / "smoke" / "history.csv")) == 3

    export = tmp_path / "model.pt"
    run(
        ROOT / "scripts/evaluate.py", "--checkpoint", best,
        "--manifest", cache / "aptos.csv", "--splits", "test",
        "--fit-manifest", cache / "aptos.csv", "--fit-splits", "val",
        "--out-dir", tmp_path / "eval", "--name", "aptos_test",
        "--export", export, "--num-workers", 0,
    )
    report = json.loads((tmp_path / "eval" / "aptos_test.json").read_text())
    assert -1 <= report["tuned_thresholds"]["qwk"] <= 1

    log = run(
        ROOT / "scripts/evaluate.py", "--checkpoint", export,
        "--manifest", cache / "idrid.csv", "--splits", "all",
        "--out-dir", tmp_path / "eval", "--name", "idrid", "--num-workers", 0,
    )
    assert "idrid: n=30" in log

    # Exported file carries everything the demo needs.
    ckpt = torch.load(export, map_location="cpu", weights_only=False)
    assert {"model", "config", "thresholds"} <= set(ckpt)
    assert ckpt["config"]["preprocess"]["size"] == 128
