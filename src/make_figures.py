"""Figures for the PrivacyLens progress report (print, light surface).

Reads the CSV outputs of build_dataset.py / run_baselines.py and writes PNGs
to results/figures/. Palette and chart chrome follow the validated reference
palette (blue #2a78d6 series-1, aqua #1baf7a series-2, blue sequential ramp).
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
BLUE = "#2a78d6"
AQUA = "#1baf7a"
SEQ_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5",
            "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Arial"],
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
    "axes.titlesize": 11, "axes.titlecolor": INK,
    "font.size": 9,
})


def style_axes(ax, xgrid=True):
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(AXIS)
    if xgrid:
        ax.xaxis.grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.tick_params(length=0)


def fig_label_distribution():
    dist = pd.read_csv(RESULTS / "label_distribution.csv")
    d = dist[dist.level == "mapped"].sort_values("segments")
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.barh(d.category, d.segments, color=BLUE, height=0.62)
    for y, v in enumerate(d.segments):
        ax.text(v + 25, y, f"{v:,}", va="center", fontsize=9, color=INK)
    ax.set_xlim(0, d.segments.max() * 1.14)
    ax.set_title("Mapped category frequency (segments, n=3,791; multi-label)",
                 loc="left", pad=10)
    ax.set_xlabel("Annotated segments")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_label_distribution.png", dpi=200)


def fig_model_comparison():
    comp = pd.read_csv(RESULTS / "model_comparison.csv")
    comp = comp.sort_values("macro_f1_mean")
    y = np.arange(len(comp))
    h = 0.34
    fig, ax = plt.subplots(figsize=(6.0, 2.9))
    ax.barh(y + h / 2 + 0.02, comp.macro_f1_mean, height=h, color=BLUE,
            label="Macro-F1")
    ax.barh(y - h / 2 - 0.02, comp.micro_f1_mean, height=h, color=AQUA,
            label="Micro-F1")
    for yi, v in zip(y + h / 2 + 0.02, comp.macro_f1_mean):
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    for yi, v in zip(y - h / 2 - 0.02, comp.micro_f1_mean):
        ax.text(v + 0.008, yi, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(y, comp.model)
    ax.set_xlim(0, 1.0)
    ax.set_title("Model comparison — 5-fold policy-grouped CV (train split)",
                 loc="left", pad=10)
    ax.set_xlabel("F1 score")
    ax.legend(frameon=False, loc="lower right", fontsize=8.5)
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_model_comparison.png", dpi=200)


def fig_per_category_f1():
    per = pd.read_csv(RESULTS / "per_category_f1.csv")
    best = per[per.model == "Linear SVM"].sort_values("f1")
    fig, ax = plt.subplots(figsize=(6.0, 2.6))
    ax.barh(best.category, best.f1, color=BLUE, height=0.62)
    for y, (v, s) in enumerate(zip(best.f1, best.support)):
        ax.text(v + 0.012, y, f"{v:.2f}  (n={s:,})", va="center",
                fontsize=8.5, color=INK)
    ax.set_xlim(0, 1.0)
    ax.set_title("Linear SVM — per-category F1 (out-of-fold CV predictions)",
                 loc="left", pad=10)
    ax.set_xlabel("F1 score")
    style_axes(ax)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_per_category_f1.png", dpi=200)


def fig_confusion():
    """Row-normalized 'confusion' on single-gold-label segments: for segments
    whose gold set is exactly one category, the share of out-of-fold
    predictions that include each category (rows can exceed 1 in principle;
    each cell is P(category predicted | gold))."""
    pred = pd.read_csv(RESULTS / "cv_predictions.csv")
    single = pred[~pred.labels.str.contains(r"\|")]
    cats = (pd.read_csv(RESULTS / "label_distribution.csv")
            .query("level == 'mapped'").category.tolist())
    M = np.zeros((len(cats), len(cats)))
    counts = np.zeros(len(cats))
    idx = {c: i for i, c in enumerate(cats)}
    for gold, p in zip(single.labels, single.predicted.fillna("")):
        gi = idx[gold]
        counts[gi] += 1
        for c in p.split("|"):
            if c:
                M[gi, idx[c]] += 1
    M = M / np.maximum(counts[:, None], 1)

    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQ_RAMP)
    fig, ax = plt.subplots(figsize=(5.6, 4.6))
    ax.imshow(M, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    short = [c.replace(" & ", " &\n") for c in cats]
    ax.set_xticks(range(len(cats)), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(cats)),
                  [f"{c}  (n={int(n):,})" for c, n in zip(cats, counts)],
                  fontsize=8)
    for i in range(len(cats)):
        for j in range(len(cats)):
            if M[i, j] >= 0.05:
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=7.5,
                        color="#ffffff" if M[i, j] > 0.55 else INK)
    ax.set_title("Linear SVM — prediction rate by gold category\n"
                 "(segments with exactly one gold label)", loc="left", pad=10)
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Gold category")
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_confusion.png", dpi=200)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    fig_label_distribution()
    fig_model_comparison()
    fig_per_category_f1()
    fig_confusion()
    print("wrote 4 figures to", FIGS)
