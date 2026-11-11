import numpy as np

from retina.metrics import (
    DEFAULT_THRESHOLDS, full_report, optimize_thresholds, qwk, referable_metrics, scores_to_grades,
)


def test_qwk_perfect_and_worst_ordering():
    y = np.array([0, 1, 2, 3, 4] * 4)
    assert qwk(y, y) == 1.0
    assert qwk(y, np.clip(y + 1, 0, 4)) > qwk(y, 4 - y)


def test_scores_to_grades():
    s = np.array([-1, 0.49, 0.5, 1.6, 3.49, 3.5, 9])
    assert scores_to_grades(s, DEFAULT_THRESHOLDS).tolist() == [0, 0, 1, 2, 3, 4, 4]


def test_optimize_thresholds_fixes_a_shifted_model():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 5, 500)
    scores = y * 0.6 + 0.3 + rng.normal(0, 0.1, 500)  # compressed, shifted scale
    before = qwk(y, scores_to_grades(scores, DEFAULT_THRESHOLDS))
    t = optimize_thresholds(scores, y)
    after = qwk(y, scores_to_grades(scores, t))
    assert after > before and after > 0.95
    assert t == sorted(t)


def test_referable_and_report():
    y = np.array([0, 0, 1, 2, 3, 4])
    s = np.array([0.1, 0.2, 1.0, 2.1, 3.0, 4.2])
    ref = referable_metrics(s, y, DEFAULT_THRESHOLDS)
    assert ref["auc"] == 1.0 and ref["sensitivity"] == 1.0 and ref["specificity"] == 1.0
    rep = full_report(s, y, DEFAULT_THRESHOLDS)
    assert rep["qwk"] == 1.0 and rep["n"] == 6
    assert np.array(rep["confusion_matrix"]).trace() == 6


def test_optimize_thresholds_ignores_nan_and_inf():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 300)
    scores = y + rng.normal(0, 0.3, 300)
    scores[[3, 10]] = [np.nan, np.inf]
    t = optimize_thresholds(scores, y)
    assert t == sorted(t) and all(np.isfinite(t))


def test_optimize_thresholds_keeps_bands_wide():
    """Few severe examples must not squeeze the severe band to a sliver."""
    rng = np.random.default_rng(2)
    y = np.repeat([0, 1, 2, 3, 4], [300, 40, 120, 6, 30])
    scores = y + rng.normal(0, 0.35, len(y))
    scores[y == 3] = 2.9 + rng.normal(0, 0.05, 6)  # severe eyes bunched up
    t = optimize_thresholds(scores, y, min_gap=0.5)
    assert all(b - a >= 0.5 - 1e-9 for a, b in zip(t, t[1:]))
    loose = optimize_thresholds(scores, y, min_gap=0.0)
    assert min(b - a for a, b in zip(loose, loose[1:])) < 0.5  # unconstrained does squeeze
