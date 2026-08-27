"""Shared grouped-bar figure layout and plotting helpers, used by the term_expansion, dataset_comparison, and lora_finetuning poster/report figure scripts."""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.artist import Artist

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.lib.analysis.plot_style import (
    COLOR_DICTIONARY,
    COLOR_EXTERNAL_DICTIONARY,
    COLOR_GPT,
    COLOR_GPT_EXPAND,
    COLOR_LORA,
    COLOR_NO_OVERLAP_DATA,
    COLOR_NO_TERM,
    COLOR_ORIGINAL,
    COLOR_OVERLAP_DATA,
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
    "dictionary": COLOR_EXTERNAL_DICTIONARY,
    "cleaned": COLOR_DICTIONARY,
    "external_dictionary": COLOR_EXTERNAL_DICTIONARY,
    "dev_v1": COLOR_ORIGINAL,
    "dev_v2": COLOR_DICTIONARY,
    "gpt_few_shot": COLOR_GPT,
    "qwen_lora_2ep": COLOR_LORA,
    "overlap_data": COLOR_OVERLAP_DATA,
    "no_overlap_data": COLOR_NO_OVERLAP_DATA,
    "zero_shot": COLOR_ORIGINAL,
    "few_shot": COLOR_GPT_EXPAND,
    "qwen_base_few_shot": COLOR_ORIGINAL,
    "qwen_lora_zero_shot": COLOR_GPT_EXPAND,
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

# Report-sized (ACL two-column-spanning figure*) variants of the constants above,
# used by report/acl_latex.tex's Experiments-section figures instead of the
# poster-sized defaults. Pass these explicitly to create_pair_figure() /
# finalize_pair_layout() / place_legend_row() -- the poster defaults above are
# left untouched for every other figure script. These figures have no suptitle
# (report/acl_latex.tex's LaTeX captions cover that); the legend is a
# figure-level, titleless, borderless, single-row legend sitting at the top of
# the figure, right above the panels.
REPORT_PAIR_FIG_SIZE = (6.8, 2.75)
REPORT_PAIR_TOP = 0.80
REPORT_PAIR_BOTTOM = 0.20
REPORT_PAIR_LEFT = 0.08
REPORT_PAIR_RIGHT = 0.98
REPORT_PAIR_WSPACE = 0.32
REPORT_LEGEND_WIDTH_RATIO = 0.26  # only used if legend_position="side"
REPORT_SUPTITLE_FONTSIZE = 11

REPORT_SIDE_LEGEND_FONTSIZE = 8.5
REPORT_SIDE_LEGEND_TITLE_FONTSIZE = 9
REPORT_SIDE_LEGEND_LABELSPACING = 0.6
REPORT_SIDE_LEGEND_BORDERPAD = 0.5

REPORT_LEGEND_ROW_FONTSIZE = 8.5
REPORT_LEGEND_ROW_NCOL = 3
REPORT_LEGEND_ROW_Y = 0.99  # bbox_to_anchor y: top of the figure (no title above it now)

REPORT_VALUE_FONTSIZE = 5.5

REPORT_GROUP_WIDTH = 0.92  # up from GROUP_WIDTH=0.86 -- wider bars, less gap between groups
REPORT_BAR_WIDTH_RATIO = 0.98  # up from BAR_WIDTH_RATIO=0.96 -- less gap between bars in a group

# Single-panel report-style figure (e.g. figure_few_shot_ablation.py). Full
# REPORT_PAIR_FIG_SIZE width, not half -- this panel has 5 two-line-wrapped
# group labels (e.g. "Weighted\nCons (%)"), which need as much horizontal
# room as a full pair figure to avoid neighboring labels colliding.
REPORT_SINGLE_FIG_SIZE = (6.8, 2.75)
REPORT_SINGLE_TOP = 0.80
REPORT_SINGLE_BOTTOM = 0.20
REPORT_SINGLE_LEFT = 0.08
REPORT_SINGLE_RIGHT = 0.98

# Vertically-stacked, single-column report-style figure (e.g.
# figure_qwen_size_comparison.py): two panels stacked top/bottom instead of
# side by side, sized to fit an ACL single-column width rather than the
# two-column-spanning REPORT_PAIR_FIG_SIZE.
REPORT_STACK_FIG_SIZE = (3.3, 5.4)
REPORT_STACK_TOP = 0.90
REPORT_STACK_BOTTOM = 0.11
REPORT_STACK_LEFT = 0.16
REPORT_STACK_RIGHT = 0.97
REPORT_STACK_HSPACE = 0.15

# 2x2-grid report-style figure (e.g. figure_few_shot_ablation_qwen.py). Wider
# than the pair figure -- each of its 4 panels independently needs the same
# 5-two-line-label horizontal room as REPORT_SINGLE_FIG_SIZE above, so the
# total width roughly doubles a single panel's.
REPORT_GRID2X2_FIG_SIZE = (10.5, 6.5)
REPORT_GRID2X2_TOP = 0.88
REPORT_GRID2X2_BOTTOM = 0.10
REPORT_GRID2X2_LEFT = 0.06
REPORT_GRID2X2_RIGHT = 0.98
REPORT_GRID2X2_WSPACE = 0.28
REPORT_GRID2X2_HSPACE = 0.55

EXP1_VARIANTS = ("original", "expand", "cleaned")
EXP23_STRATEGIES = ("original", "expand", "cleaned")


def ylim_from_values(values: list[float | None], pad_ratio: float = 0.14) -> tuple[float, float]:
    present = [v for v in values if v is not None]
    if not present:
        return (0.0, 1.0)
    return (0.0, max(present) * (1 + pad_ratio))


def compute_shared_expansion_ylims(results_root: Path) -> dict[str, tuple[float, float]]:
    """Shared BLEU / term-accuracy y-limits across model_comparison and mode_comparison for visual comparison."""
    from shared.lib.analysis.metrics_loader import (
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
    value_fontsize: float = VALUE_FONTSIZE,
    panel_title_fontsize: float = 14,
    show_values: bool = True,
    group_width: float = GROUP_WIDTH,
    bar_width_ratio: float = BAR_WIDTH_RATIO,
) -> list[float | None]:
    n_series = len(series_keys)
    x = np.arange(len(group_keys))
    bar_width = group_width / n_series
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
            bar_width * bar_width_ratio,
            label=series_labels[series_key],
            color=EXPANSION_COLORS[series_key],
            edgecolor=BAR_EDGE,
            linewidth=BAR_LW,
        )
        if show_values:
            for bar, val in zip(bars, values):
                if val is not None:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        val,
                        f"{val:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=value_fontsize,
                        fontweight="bold",
                    )

    if panel_title:
        ax.set_title(panel_title, loc="left", fontsize=panel_title_fontsize, color=POSTER_BLUE, pad=12)
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


def create_pair_figure(
    suptitle: str | None,
    *,
    figsize: tuple[float, float] = PAIR_FIG_SIZE,
    legend_width_ratio: float = LEGEND_WIDTH_RATIO,
    wspace: float = PAIR_WSPACE,
    hspace: float | None = None,
    suptitle_fontsize: float = 16,
    legend_position: str = "side",
    stacked: bool = False,
) -> tuple[Figure, np.ndarray, Axes | None]:
    """``legend_position="side"`` (default, used by every poster figure script):
    2-column layout, panels | narrow legend column; returns a legend Axes to
    pass to place_side_legend().
    ``legend_position="bottom"`` (used by the report/acl_latex.tex figures):
    panels only; a figure-level legend is placed separately by
    place_legend_row(fig, ...), typically at the top of the figure -- more
    robust than a dedicated legend-row Axes, which doesn't reliably reserve
    enough space for a multi-row legend. Returns ``None`` in place of a legend
    Axes; the caller uses ``fig`` instead.
    ``stacked=True`` (used by the report/acl_latex.tex figures, single-column
    width): the two panels are arranged 2x1 (top/bottom) instead of 1x2
    (side by side), spaced by ``hspace`` instead of ``wspace``. Only valid
    with ``legend_position="bottom"``.
    ``suptitle=None`` skips the title entirely (used by the report figures,
    whose LaTeX captions already cover what the title would have said).
    """
    fig = plt.figure(figsize=figsize)
    if legend_position == "bottom":
        if stacked:
            axes = fig.subplots(2, 1)
            if hspace is not None:
                fig.subplots_adjust(hspace=hspace)
        else:
            axes = fig.subplots(1, 2)
            fig.subplots_adjust(wspace=wspace)
        legend_ax = None
    else:
        gs = fig.add_gridspec(1, 2, width_ratios=[1, legend_width_ratio], wspace=0.08)
        gs_axes = gs[0].subgridspec(1, 2, wspace=wspace)
        axes = np.array([fig.add_subplot(gs_axes[0, 0]), fig.add_subplot(gs_axes[0, 1])])
        legend_ax = fig.add_subplot(gs[1])
        legend_ax.axis("off")
    if suptitle is None:
        return fig, axes, legend_ax
    fig.suptitle(suptitle, fontsize=suptitle_fontsize, fontweight="bold", color=POSTER_BLUE, y=0.98)
    return fig, axes, legend_ax


def place_side_legend(
    legend_ax: Axes,
    handles: list[Artist],
    title: str,
    *,
    fontsize: float = SIDE_LEGEND_FONTSIZE,
    title_fontsize: float = SIDE_LEGEND_TITLE_FONTSIZE,
    labelspacing: float = SIDE_LEGEND_LABELSPACING,
    borderpad: float = SIDE_LEGEND_BORDERPAD,
) -> None:
    legend_ax.legend(
        handles=handles,
        loc="center",
        ncol=1,
        framealpha=0.95,
        title=title,
        title_fontsize=title_fontsize,
        fontsize=fontsize,
        labelspacing=labelspacing,
        borderpad=borderpad,
    )


def place_legend_row(
    fig: Figure,
    handles: list[Artist],
    *,
    fontsize: float = REPORT_LEGEND_ROW_FONTSIZE,
    ncol: int = REPORT_LEGEND_ROW_NCOL,
    y: float = REPORT_LEGEND_ROW_Y,
    loc: str = "upper center",
) -> None:
    """Figure-level, horizontal, titleless, borderless, possibly-multi-row
    legend -- used with create_pair_figure(..., legend_position="bottom").
    Default ``y``/``loc`` place it between the suptitle and the panels; a
    figure-level legend (rather than a dedicated legend-row Axes) reliably
    reserves its own space regardless of how many rows it wraps to."""
    ncol = min(ncol, len(handles)) or 1
    fig.legend(
        handles=handles,
        loc=loc,
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=False,
        fontsize=fontsize,
        columnspacing=1.2,
        handletextpad=0.5,
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


def finalize_pair_layout(
    fig: Figure,
    *,
    top: float = PAIR_TOP,
    bottom: float = PAIR_BOTTOM,
    left: float = PAIR_LEFT,
    right: float = PAIR_RIGHT,
) -> None:
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right)


def finalize_stack_layout(
    fig: Figure,
    *,
    top: float = REPORT_STACK_TOP,
    bottom: float = REPORT_STACK_BOTTOM,
    left: float = REPORT_STACK_LEFT,
    right: float = REPORT_STACK_RIGHT,
    hspace: float = REPORT_STACK_HSPACE,
) -> None:
    fig.subplots_adjust(top=top, bottom=bottom, left=left, right=right, hspace=hspace)
