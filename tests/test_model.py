import torch

from retina.gradcam import gradcam
from retina.model import RetinaModel, load_weights


def _model(head="regression"):
    return RetinaModel("resnet18", head=head, pretrained=False, drop_path=0.0).eval()


def test_regression_and_classification_outputs():
    x = torch.randn(2, 3, 96, 96)
    reg = _model("regression")
    assert reg(x).shape == (2,)
    cls = _model("classification")
    out = cls(x)
    assert out.shape == (2, 5)
    s = cls.to_score(out)
    assert ((s >= 0) & (s <= 4)).all()


def test_gradcam_shape_and_range():
    cam = gradcam(_model(), torch.randn(1, 3, 96, 96))
    assert cam.shape == (96, 96)
    assert cam.min() >= 0 and cam.max() <= 1


def test_load_weights_skips_mismatched_head(tmp_path):
    src = _model("classification")
    torch.save({"model": src.state_dict()}, tmp_path / "c.pt")
    dst = _model("regression")
    load_weights(dst, tmp_path / "c.pt")
    w_src = src.backbone.conv1.weight
    assert torch.equal(dst.backbone.conv1.weight, w_src)
    assert dst.fc.weight.shape == (1, 512)
