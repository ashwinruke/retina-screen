"""Make the README figures from a training run and an evaluation.

    python scripts/plot_results.py \
        --history outputs/baseline_aptos/history.csv \
        --report outputs/eval/mvp_aptos_test.json \
        --predictions outputs/eval/mvp_aptos_test_predictions.csv \
        --out-dir docs/figures --prefix mvp

Writes <prefix>_training.png, <prefix>_confusion.png and <prefix>_roc.png.
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from sklearn.metrics import roc_curve  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
SERIES_1 = "#2a78d6"  # blue
SERIES_2 = "#eb6834"  # orange
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#fcfcfb", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]
)
GRADES = ["0 No DR", "1 Mild", "2 Moderate", "3 Severe", "4 Prolif."]


def _style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel(xlabel, color=INK_2, fontsize=9)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _figure(ncols: int = 1, width: float = 5.0):
    fig, axes = plt.subplots(1, ncols, figsize=(width * ncols, 3.6), squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes[0]


def plot_training(history: pd.DataFrame, path: Path) -> None:
    """Two panels, never one dual-axis chart: loss and QWK have different scales."""
    fig, (ax_loss, ax_qwk) = _figure(2, 4.6)
    epochs = history["epoch"] + 1

    ax_loss.plot(epochs, history["train_loss"], color=SERIES_1, linewidth=2)
    _style(ax_loss, "Training loss", "Epoch", "SmoothL1 loss")
    ax_loss.set_ylim(bottom=0)

    ax_qwk.plot(epochs, history["val_qwk_tuned"], color=SERIES_1, linewidth=2, label="Tuned thresholds")
    ax_qwk.plot(epochs, history["val_qwk"], color=SERIES_2, linewidth=2, label="Fixed thresholds")
    best = history["val_qwk_tuned"].idxmax()
    ax_qwk.plot(epochs[best], history["val_qwk_tuned"][best], "o", color=SERIES_1,
                markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    ax_qwk.annotate(f"best {history['val_qwk_tuned'][best]:.3f}", (epochs[best], history["val_qwk_tuned"][best]),
                    xytext=(0, -16), textcoords="offset points", ha="center", color=INK, fontsize=8)
    _style(ax_qwk, "Validation QWK", "Epoch", "Quadratic weighted kappa")
    ax_qwk.legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower right")
    for ax in (ax_loss, ax_qwk):
        ax.set_xlim(0.5, epochs.max() + 0.5)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_confusion(report: dict, path: Path) -> None:
    cm = np.array(report["confusion_matrix"], dtype=float)
    share = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, (ax,) = _figure(1, 5.0)
    fig.set_size_inches(5.2, 4.4)
    ax.imshow(share, cmap=BLUES, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if cm[i, j] == 0:
                continue
            ax.text(j, i, f"{int(cm[i, j])}\n{share[i, j]:.0%}", ha="center", va="center", fontsize=8,
                    color=SURFACE if share[i, j] > 0.55 else INK)
    ax.set_xticks(range(5), GRADES, rotation=30, ha="right")
    ax.set_yticks(range(5), GRADES)
    ax.set_title(f"Confusion matrix (n={report['n']}, QWK {report['qwk']:.3f})",
                 loc="left", color=INK, fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Predicted grade", color=INK_2, fontsize=9)
    ax.set_ylabel("True grade (row % = recall)", color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=8, length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_roc(preds: pd.DataFrame, report: dict, path: Path) -> None:
    target = preds["grade"] >= 2
    fpr, tpr, _ = roc_curve(target, preds["score"])
    ref = report["referable"]
    fig, (ax,) = _figure(1, 4.4)
    fig.set_size_inches(4.4, 4.2)
    ax.plot([0, 1], [0, 1], color=GRID, linewidth=1.5, linestyle="--")
    ax.plot(fpr, tpr, color=SERIES_1, linewidth=2)
    op = (1 - ref["specificity"], ref["sensitivity"])
    ax.plot(*op, "o", color=SERIES_1, markersize=8, markeredgecolor=SURFACE, markeredgewidth=2)
    # Put the label toward the middle of the plot so it never hits an axis.
    dx = 14 if op[0] < 0.5 else -14
    dy = -30 if op[1] > 0.4 else 24
    ax.annotate(f"operating point\nsens {ref['sensitivity']:.2f}, spec {ref['specificity']:.2f}", op,
                xytext=(dx, dy), textcoords="offset points", color=INK, fontsize=8,
                ha="left" if dx > 0 else "right")
    _style(ax, f"Referable DR (grade ≥ 2), AUC {ref['auc']:.3f}", "False positive rate", "Sensitivity")
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--report", required=True, help="JSON written by evaluate.py")
    ap.add_argument("--predictions", required=True, help="*_predictions.csv written by evaluate.py")
    ap.add_argument("--out-dir", default="docs/figures")
    ap.add_argument("--prefix", default="mvp")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    report = json.loads(Path(args.report).read_text())["tuned_thresholds"]
    plot_training(pd.read_csv(args.history), out / f"{args.prefix}_training.png")
    plot_confusion(report, out / f"{args.prefix}_confusion.png")
    plot_roc(pd.read_csv(args.predictions), report, out / f"{args.prefix}_roc.png")
    print("Figures written to", out)


if __name__ == "__main__":
    main()
