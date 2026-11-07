"""The in-browser model must give the same answers as the Python pipeline."""
import numpy as np
import pytest
import torch

from retina.export import BrowserModel, export_onnx
from retina.gradcam import gradcam
from retina.model import RetinaModel
from retina.preprocess import crop_and_resize, preprocess_fundus

SIZE = 128


def _fundus(seed: int) -> np.ndarray:
    import sys

    from conftest import ROOT

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_synthetic import fake_fundus

    return fake_fundus(seed % 5, np.random.default_rng(seed), 300, 400)


def _model() -> RetinaModel:
    torch.manual_seed(0)
    m = RetinaModel("resnet18", pretrained=False, drop_path=0.0)
    with torch.no_grad():
        m.fc.weight.normal_(0, 0.5)
    return m.eval()


def test_ben_graham_in_graph_matches_opencv():
    bm = BrowserModel(_model(), {"size": SIZE})
    for seed in range(3):
        img = _fundus(seed)
        ref = preprocess_fundus(img, SIZE).astype(np.float32)
        x = torch.from_numpy(crop_and_resize(img, SIZE).astype(np.float32).transpose(2, 0, 1))[None]
        got = bm.ben_graham(x)[0].numpy().transpose(1, 2, 0)
        diff = np.abs(got - ref)
        assert diff.max() <= 1.0  # float rounding may flip a value by one grey level
        assert (diff > 0).mean() < 0.01


def test_closed_form_gradcam_matches_autograd():
    model = _model()
    bm = BrowserModel(model, {"size": SIZE, "ben_graham": False})
    x = torch.rand(1, 3, SIZE, SIZE) * 255
    _, cam, seen = bm(x)
    xn = (seen - bm.mean) / bm.std
    ref = gradcam(model, xn)
    assert np.abs(cam[0].detach().numpy() - ref).max() < 1e-4


def test_onnx_export_matches_pytorch(tmp_path):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    model = _model()
    cfg = {"preprocess": {"size": SIZE, "ben_graham": True}, "model": {"backbone": "resnet18"}}
    path = export_onnx(model, cfg, [0.5, 1.5, 2.5, 3.5], tmp_path)
    assert (tmp_path / "model.json").exists()

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    bm = BrowserModel(model, cfg["preprocess"]).eval()
    x = torch.from_numpy(crop_and_resize(_fundus(7), SIZE).astype(np.float32).transpose(2, 0, 1))[None]
    with torch.no_grad():
        score, cam, seen = bm(x)
    o_score, o_cam, o_seen = sess.run(None, {"image": x.numpy()})
    assert abs(float(o_score[0]) - float(score[0])) < 1e-3
    assert np.abs(o_cam - cam.numpy()).max() < 1e-3
    assert np.abs(o_seen - seen.numpy()).max() <= 1.0
