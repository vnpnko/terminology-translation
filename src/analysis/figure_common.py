"""Shared layout and grouped-bar helpers for paired expansion figures (model_comparison & mode_comparison)."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.artist import Artist

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.plot_style import (
    COLOR_DICTIONARY,
    COLOR_EXTERNAL_DICTIONARY,
    COLOR_GPT_EXPAND,
    COLOR_NO_TERM,
    COLOR_ORIGINAL,
    COLOR_RANDOM_TERM,
    POSTER_BLUE,
    fixed_top_ylim,
    headroom_ylim,
)

BLEU_YLIM_TOP = 70
TERM_ACC_YLIM_TOP = 100
YLIM_LABEL_PAD = 0.06

# Shared colors for the two overlapping variants (original, expand) and each third arm.
EXPANSION_COLORS = {
    "no_term": COLOR_NO_TERM,
    "random_term": COLOR_RANDOM_TERM,
    "original": COLOR_ORIGINAL,
    "expand": COLOR_GPT_EXPAND,
    "dictionary": COLOR_DICTIONARY,
    "cleaned": COLOR_DICTIONARY,
    "external_dictionary": COLOR_EXTERNAL_DICTIONARY,
    "dev_v1": COLOR_ORIGINAL,
    "dev_v2": COLOR_DICTIONARY,
}

PAIR_FIG_SIZE = (19, 6.5)
PAIR_TOP = 0.86
PAIR_BOTTOM = 0.17
PAIR_LEFT = 0.07
PAIR_RIGHT = 0.98
PAIR_WSPACE = 0.22
LEGEND_WIDTH_RATIO = 0.15

SIDE_LEGEND_FONTSIZE = 13
SIDE_LEGEND_TITLE_FONTSIZE = 14
SIDE_LEGEND_LABELSPACING = 0.9
SIDE_LEGEND_BORDERPAD = 0.8

CAPTION_FONTSIZE = 12
CAPTION_Y = 0.018
CAPTION_X = 0.44

GROUP_WIDTH = 0.86
BAR_EDGE = "#333333"
BAR_LW = 0.6
BAR_WIDTH_RATIO = 0.96
VALUE_FONTSIZE = 9

EXP1_VARIANTS = ("original", "expand", "cleaned")
EXP23_STRATEGIES = ("original", "expand", "cleaned")


def ylim_from_values(values: list[float | None], pad_ratio: float = 0.14) -> tuple[float, float]:
    present = [v for v in values if v is not None]
    if not present:
        return (0.0, 1.0)
    return (0.0, max(present) * (1 + pad_ratio))


def compute_shared_expansion_ylims(results_root: Path) -> dict[str, tuple[float, float]]:
    """Shared BLEU / term-accuracy y-limits across model_comparison and mode_comparison for visual comparison."""
    from src.analysis.metrics_loader import (
        BASELINE_DIRS,
        DEFAULT_MODE,
        LANG_ORDER,
        get_lang_mode_metrics,
        load_metrics_path,
        macro_average,
    )

    bleu_values: list[float | None] = []
    term_values: list[float | None] = []

    for variant in EXP1_VARIANTS:
        for baseline in BASELINE_DIRS:
            summary = load_metrics_path(
                results_root / "dev_v1" / variant / baseline / "metrics_summary.json"
            )
            row = macro_average(summary, DEFAULT_MODE)
            bleu_values.append(row.get("bleu"))
            term_values.append(row.get("term_accuracy_pct"))

    for strategy in EXP23_STRATEGIES:
        summary = load_metrics_path(
            results_root / "dev_v1" / strategy / "gpt" / "metrics_summary.json"
        )
        for lang in LANG_ORDER:
            row = get_lang_mode_metrics(summary, lang, DEFAULT_MODE)
            bleu_values.append(row.get("bleu"))
            term_values.append(row.get("term_accuracy_pct"))

    return {
        "bleu": ylim_from_values(bleu_values),
        "term_accuracy_pct": ylim_from_values(term_values),
    }


def plot_grouped_bars(
    ax: Axes,
    *,
    group_keys: tuple[str, ...],
    group_labels: dict[str, str] | list[str],
    series_keys: tuple[str, ...],
    series_labels: dict[str, str],
    value_fn: Callable[[str, str], float | None],
    ylabel: str,
    panel_title: str | None = None,
    xlabel: str | None = None,
    ylim_top: float | None = None,
    yticks: list[float] | np.ndarray | None = None,
) -> list[float | None]:
    n_series = len(series_keys)
    x = np.arange(len(group_keys))
    bar_width = GROUP_WIDTH / n_series
    all_values: list[float | None] = []

    if isinstance(group_labels, dict):
        tick_labels = [group_labels[k] for k in group_keys]
    else:
        tick_labels = list(group_labels)

    for s_idx, series_key in enumerate(series_keys):
        offset = (s_idx - (n_series - 1) / 2) * bar_width
        values = [value_fn(series_key, group_key) for group_key in group_keys]
        all_values.extend(values)
        # Missing values (e.g. term accuracy for the no-term baseline) render as
        # empty bars instead of raising in matplotlib.
        plot_values = [np.nan if v is None else v for v in values]
        bars = ax.bar(
            x + offset,
            plot_values,
            bar_width * BAR_WIDTH_RATIO,
            label=series_labels[series_key],
            color=EXPANSION_COLORS[series_key],
            edgecolor=BAR_EDGE,
            linewidth=BAR_LW,
        )
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=VALUE_FONTSIZE,
                    fontweight="bold",
                )

    if panel_title:
        ax.set_title(panel_title, loc="left", fontsize=14, color=POSTER_BLUE, pad=12)
    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels)
    ax.set_ylabel(ylabel, labelpad=10)
    if xlabel:
        ax.set_xlabel(xlabel, labelpad=10)
    if ylim_top is not None:
        fixed_top_ylim(ax, ylim_top, pad_ratio=YLIM_LABEL_PAD)
    else:
        headroom_ylim(ax, all_values)
    if yticks is not None:
        ax.set_yticks(yticks)
    return all_values


def create_pair_figure(suptitle: str) -> tuple[Figure, np.ndarray, Axes]:
    fig = plt.figure(figsize=PAIR_FIG_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, LEGEND_WIDTH_RATIO], wspace=0.08)
    gs_axes = gs[0].subgridspec(1, 2, wspace=PAIR_WSPACE)
    axes = np.array([fig.add_subplot(gs_axes[0, 0]), fig.add_subplot(gs_axes[0, 1])])
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis("off")
    fig.suptitle(suptitle, fontsize=16, fontweight="bold", color=POSTER_BLUE, y=0.98)
    return fig, axes, legend_ax


def place_side_legend(
    legend_ax: Axes,
    handles: list[Artist],
    title: str,
) -> None:
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        framealpha=0.95,
        title=title,
        title_fontsize=SIDE_LEGEND_TITLE_FONTSIZE,
        fontsize=SIDE_LEGEND_FONTSIZE,
        labelspacing=SIDE_LEGEND_LABELSPACING,
        borderpad=SIDE_LEGEND_BORDERPAD,
    )


def place_figure_caption(fig: Figure, text: str) -> None:
    fig.text(
        CAPTION_X,
        CAPTION_Y,
        text,
        ha="center",
        va="bottom",
        fontsize=CAPTION_FONTSIZE,
        color="#444444",
    )


def finalize_pair_layout(fig: Figure) -> None:
    fig.subplots_adjust(
        top=PAIR_TOP,
        bottom=PAIR_BOTTOM,
        left=PAIR_LEFT,
        right=PAIR_RIGHT,
    )
