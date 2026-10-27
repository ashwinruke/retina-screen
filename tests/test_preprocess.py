import numpy as np

from retina.preprocess import (
    BEN_BACKGROUND, crop_to_fundus, pad_to_square, preprocess_fundus,
)


def _fundus(h=300, w=400, border=40):
    img = np.zeros((h, w, 3), np.uint8)
    img[border : h - border, border + 20 : w - border - 20] = (200, 90, 40)
    return img


def test_crop_removes_black_border():
    out = crop_to_fundus(_fundus())
    assert out.shape[:2] == (220, 280)
    assert out.min() > 0


def test_crop_leaves_black_image_alone():
    img = np.zeros((50, 60, 3), np.uint8)
    assert crop_to_fundus(img).shape == img.shape


def test_pad_to_square_centres():
    out = pad_to_square(np.full((10, 30, 3), 255, np.uint8))
    assert out.shape == (30, 30, 3)
    assert out[0, 0].sum() == 0 and out[15, 15].sum() == 765


def test_preprocess_shapes_and_background():
    out = preprocess_fundus(_fundus(), size=128, use_ben_graham=True)
    assert out.shape == (128, 128, 3) and out.dtype == np.uint8
    assert (out[0, 0] == BEN_BACKGROUND).all()  # outside the circular mask

    plain = preprocess_fundus(_fundus(), size=64, use_ben_graham=False)
    assert plain.shape == (64, 64, 3)


def test_no_edge_bands_on_clipped_fundus():
    """A uniformly coloured retina clipped at top and bottom should come out
    flat grey: no bright/dark bands where it meets the black background."""
    import cv2

    from retina.preprocess import ben_graham, crop_and_resize

    img = np.zeros((600, 900, 3), np.uint8)
    cv2.circle(img, (450, 300), 380, (200, 90, 40), -1)  # taller than the frame
    out = preprocess_fundus(img, size=256)
    assert np.abs(out.astype(int) - BEN_BACKGROUND).max() <= 3

    # The plain version does produce bands, which is what this guards against.
    plain = ben_graham(crop_and_resize(img, 256))
    assert np.abs(plain[20:60, 128].astype(int) - BEN_BACKGROUND).max() > 30
