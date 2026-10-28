"""Metrics for ordinal DR grading and referable-DR screening."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.metrics import cohen_kappa_score, confusion_matrix, roc_auc_score

NUM_GRADES = 5
DEFAULT_THRESHOLDS = [0.5, 1.5, 2.5, 3.5]
REFERABLE_GRADE = 2  # grade >= 2 (moderate NPDR or worse) means refer

GRADE_NAMES = [
    "No DR",
    "Mild NPDR",
    "Moderate NPDR",
    "Severe NPDR",
    "Proliferative DR",
]


def qwk(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    """Quadratic weighted kappa over the five grades."""
    return float(
        cohen_kappa_score(
            y_true, y_pred, weights="quadratic", labels=list(range(NUM_GRADES))
        )
    )


def scores_to_grades(scores: np.ndarray, thresholds: Sequence[float]) -> np.ndarray:
    """Cut continuous severity scores into integer grades 0..4."""
    return np.digitize(np.asarray(scores, dtype=float), np.asarray(thresholds)).astype(int)


def optimize_thresholds(
    scores: np.ndarray,
    y_true: np.ndarray,
    init: Sequence[float] = DEFAULT_THRESHOLDS,
    passes: int = 4,
    steps: int = 41,
) -> list[float]:
    """Tune the four cut points to maximise QWK by coordinate search.

    Each threshold is searched on a grid between its neighbours, one at a
    time, for a few passes. Fit this on validation data only.
    """
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("All scores are NaN/inf; cannot fit thresholds.")
    scores, y_true = scores[finite], y_true[finite]
    t = list(map(float, init))
    best = qwk(y_true, scores_to_grades(scores, t))
    for _ in range(passes):
        improved = False
        for i in range(len(t)):
            lo = t[i - 1] + 1e-3 if i > 0 else min(scores.min(), t[i]) - 0.5
            hi = t[i + 1] - 1e-3 if i < len(t) - 1 else max(scores.max(), t[i]) + 0.5
            for cand in np.linspace(lo, hi, steps):
                trial = t.copy()
                trial[i] = float(cand)
                k = qwk(y_true, scores_to_grades(scores, trial))
                if k > best + 1e-9:
                    best, t, improved = k, trial, True
        if not improved:
            break
    return sorted(t)


def referable_metrics(
    scores: np.ndarray, y_true: np.ndarray, thresholds: Sequence[float]
) -> dict:
    """Binary screening metrics: is the eye grade >= 2?"""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    target = y_true >= REFERABLE_GRADE
    pred = scores_to_grades(scores, thresholds) >= REFERABLE_GRADE
    out: dict = {"n_positive": int(target.sum()), "n_negative": int((~target).sum())}
    if target.any() and (~target).any():
        out["auc"] = float(roc_auc_score(target, scores))
    else:
        out["auc"] = None
    out["sensitivity"] = float(pred[target].mean()) if target.any() else None
    out["specificity"] = float((~pred[~target]).mean()) if (~target).any() else None
    return out


def full_report(
    scores: np.ndarray, y_true: np.ndarray, thresholds: Sequence[float]
) -> dict:
    """Everything the README results table needs, as a JSON-able dict."""
    scores = np.asarray(scores, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = scores_to_grades(scores, thresholds)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_GRADES)))
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(cm.sum(1) > 0, cm.diagonal() / cm.sum(1), np.nan)
    return {
        "n": int(len(y_true)),
        "qwk": qwk(y_true, y_pred),
        "accuracy": float((y_pred == y_true).mean()),
        "thresholds": [float(x) for x in thresholds],
        "confusion_matrix": cm.tolist(),
        "per_grade_recall": {
            GRADE_NAMES[i]: (None if np.isnan(r) else float(r)) for i, r in enumerate(recall)
        },
        "referable": referable_metrics(scores, y_true, thresholds),
    }
