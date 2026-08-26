"""Terminology expansion strategies by language pair, one figure per model (GPT/Qwen 3B/Qwen 7B)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.lib.analysis.grouped_bar_figure_common import (
    BLEU_YLIM_TOP,
    EXPANSION_COLORS,
    REPORT_BAR_WIDTH_RATIO,
    REPORT_GROUP_WIDTH,
    REPORT_LEGEND_ROW_FONTSIZE,
    REPORT_LEGEND_ROW_NCOL,
    REPORT_PAIR_BOTTOM,
    REPORT_PAIR_FIG_SIZE,
    REPORT_PAIR_LEFT,
    REPORT_PAIR_RIGHT,
    REPORT_PAIR_TOP,
    REPORT_PAIR_WSPACE,
    REPORT_SUPTITLE_FONTSIZE,
    REPORT_VALUE_FONTSIZE,
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_pair_layout,
    fixed_top_ylim,
    place_legend_row,
    plot_grouped_bars,
)
from shared.lib.analysis.metrics_loader import (
    BASELINE_DIRS,
    DEFAULT_MODE,
    LANG_LABELS,
    LANG_ORDER,
    NO_TERM_MODE,
    RANDOM_TERM_MODE,
    get_lang_mode_metrics,
    load_metrics_path,
    require_paths,
)
from shared.lib.analysis.plot_style import apply_report_style

NO_TERM_KEY = "no_term"
RANDOM_TERM_KEY = "random_term"
BASELINE_SOURCE_STRATEGY = "original"
NO_TERM_LABEL = "No term"
RANDOM_TERM_LABEL = "Random term"

# "dictionary" reads shared/results/dev_v1/dictionary/ (the real term-list
# variant built by experiments/term_expansion/dictionary/'s
# build_term_dictionary.py/apply_dictionary_to_dev_v1.py) via the same
# _strategy_dir() fallback used for expand/cleaned below -- not the unrelated
# dev_v2-on-its-own-terms comparison the old EXTERNAL_DICTIONARY_* constants
# pointed at.
STRATEGY_ORDER = ("original", "expand", "cleaned", "dictionary")
SERIES_ORDER = (NO_TERM_KEY, RANDOM_TERM_KEY, *STRATEGY_ORDER)
STRATEGY_LABELS = {
    NO_TERM_KEY: NO_TERM_LABEL,
    RANDOM_TERM_KEY: RANDOM_TERM_LABEL,
    "original": "Original",
    "expand": "GPT-expanded",
    "cleaned": "GPT-cleaned",
    "dictionary": "Dictionary",
}


def _strategy_dir(results_root: Path, strategy: str) -> Path:
    if strategy == BASELINE_SOURCE_STRATEGY:
        return results_root / "dev_v1" / strategy / "few_shot"
    return results_root / "dev_v1" / strategy


def _collect_data(
    results_root: Path,
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    paths = [
        _strategy_dir(results_root, strategy) / baseline / "metrics_summary.json"
        for strategy in STRATEGY_ORDER
        for baseline in BASELINE_DIRS
    ]
    require_paths(paths)

    data: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {
        baseline: {} for baseline in BASELINE_DIRS
    }
    for strategy in STRATEGY_ORDER:
        for baseline in BASELINE_DIRS:
            summary = load_metrics_path(
                _strategy_dir(results_root, strategy) / baseline / "metrics_summary.json"
            )
            data[baseline][strategy] = {
                lang: get_lang_mode_metrics(summary, lang, DEFAULT_MODE) for lang in LANG_ORDER
            }
            if strategy == BASELINE_SOURCE_STRATEGY:
                data[baseline][NO_TERM_KEY] = {
                    lang: get_lang_mode_metrics(summary, lang, NO_TERM_MODE)
                    for lang in LANG_ORDER
                }
                data[baseline][RANDOM_TERM_KEY] = {
                    lang: get_lang_mode_metrics(summary, lang, RANDOM_TERM_MODE)
                    for lang in LANG_ORDER
                }

    return data


def build_by_language_pair_figure(results_root: Path, baseline: str) -> Figure:
    apply_report_style()
    data = _collect_data(results_root)[baseline]

    fig, axes, _ = create_pair_figure(
        None,
        figsize=REPORT_PAIR_FIG_SIZE,
        wspace=REPORT_PAIR_WSPACE,
        legend_position="bottom",
    )

    metric_axes = [
        (axes[0], "bleu", "BLEU", BLEU_YLIM_TOP, list(range(0, BLEU_YLIM_TOP + 1, 20))),
        (
            axes[1],
            "term_accuracy_pct",
            "Term accuracy (%)",
            TERM_ACC_YLIM_TOP,
            list(range(0, TERM_ACC_YLIM_TOP + 1, 25)),
        ),
    ]
    for ax, metric_key, ylabel, ylim_top, yticks in metric_axes:
        plot_grouped_bars(
            ax,
            group_keys=LANG_ORDER,
            group_labels=LANG_LABELS,
            series_keys=SERIES_ORDER,
            series_labels=STRATEGY_LABELS,
            value_fn=lambda strategy, lang, mk=metric_key: data[strategy][lang].get(mk),
            ylabel=ylabel,
            ylim_top=ylim_top,
            yticks=yticks,
            value_fontsize=REPORT_VALUE_FONTSIZE,
            panel_title_fontsize=REPORT_SUPTITLE_FONTSIZE,
            show_values=False,
            group_width=REPORT_GROUP_WIDTH,
            bar_width_ratio=REPORT_BAR_WIDTH_RATIO,
        )

    legend_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=STRATEGY_LABELS[k])
        for k in SERIES_ORDER
    ]
    place_legend_row(
        fig,
        legend_handles,
        fontsize=REPORT_LEGEND_ROW_FONTSIZE,
        ncol=REPORT_LEGEND_ROW_NCOL,
    )
    finalize_pair_layout(
        fig,
        top=REPORT_PAIR_TOP,
        bottom=REPORT_PAIR_BOTTOM,
        left=REPORT_PAIR_LEFT,
        right=REPORT_PAIR_RIGHT,
    )
    return fig


def build_by_language_pair_figures(results_root: Path) -> dict[str, Figure]:
    return {baseline: build_by_language_pair_figure(results_root, baseline) for baseline in BASELINE_DIRS}


def _lighten(hex_color: str, factor: float = 0.55) -> str:
    """Blend a hex color toward white by `factor` (0 = unchanged, 1 = white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# Appendix-only figure: BLEU and term accuracy by language pair, combining both
# Qwen sizes into stacked bars instead of two separate per-model figures.
# Verified (see report Appendix A) that Qwen2.5-7B's score is >= 3B's in every
# cell except one near-tie (cleaned/EN-DE term accuracy, 7B ~0.5pp lower) --
# each bar's bottom segment is therefore 3B's actual value and the top segment
# is a real, non-arbitrary quantity: the increment needed to reach 7B's score.
# The one near-tie is handled generically via max(0, ...) (renders as a bare
# 3B-only bar for that one cell) and explained in the appendix prose, not
# special-cased here.
def build_qwen_size_stacked_figure(results_root: Path) -> Figure:
    apply_report_style()
    data = _collect_data(results_root)
    d3, d7 = data["qwen_3b"], data["qwen_7b"]

    fig, axes = plt.subplots(1, 2, figsize=REPORT_PAIR_FIG_SIZE)
    fig.subplots_adjust(wspace=REPORT_PAIR_WSPACE)

    n_series = len(SERIES_ORDER)
    bar_width = REPORT_GROUP_WIDTH / n_series
    x = np.arange(len(LANG_ORDER))

    metric_axes = [
        (axes[0], "bleu", "BLEU", BLEU_YLIM_TOP, list(range(0, BLEU_YLIM_TOP + 1, 20))),
        (
            axes[1],
            "term_accuracy_pct",
            "Term accuracy (%)",
            TERM_ACC_YLIM_TOP,
            list(range(0, TERM_ACC_YLIM_TOP + 1, 25)),
        ),
    ]
    for ax, metric_key, ylabel, ylim_top, yticks in metric_axes:
        for s_idx, cond in enumerate(SERIES_ORDER):
            offset = (s_idx - (n_series - 1) / 2) * bar_width
            base_vals = [d3[cond][lang].get(metric_key) for lang in LANG_ORDER]
            top_vals = [d7[cond][lang].get(metric_key) for lang in LANG_ORDER]
            base_plot = [np.nan if v is None else v for v in base_vals]
            increments = [
                None if (b is None or t is None) else max(0.0, t - b)
                for b, t in zip(base_vals, top_vals)
            ]
            ax.bar(
                x + offset,
                base_plot,
                bar_width * REPORT_BAR_WIDTH_RATIO,
                color=EXPANSION_COLORS[cond],
                edgecolor="#333333",
                linewidth=0.5,
            )
            ax.bar(
                x + offset,
                [np.nan if v is None else v for v in increments],
                bar_width * REPORT_BAR_WIDTH_RATIO,
                bottom=base_plot,
                color=_lighten(EXPANSION_COLORS[cond]),
                edgecolor="#333333",
                linewidth=0.5,
            )
        ax.set_xticks(x)
        ax.set_xticklabels([LANG_LABELS[lang] for lang in LANG_ORDER])
        ax.set_ylabel(ylabel, labelpad=10)
        fixed_top_ylim(ax, ylim_top)
        ax.set_yticks(yticks)

    condition_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=STRATEGY_LABELS[k])
        for k in SERIES_ORDER
    ]
    place_legend_row(
        fig,
        condition_handles,
        fontsize=REPORT_LEGEND_ROW_FONTSIZE,
        ncol=REPORT_LEGEND_ROW_NCOL,
    )
    finalize_pair_layout(
        fig,
        top=REPORT_PAIR_TOP - 0.06,
        bottom=REPORT_PAIR_BOTTOM,
        left=REPORT_PAIR_LEFT,
        right=REPORT_PAIR_RIGHT,
    )
    return fig
