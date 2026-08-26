"""Qwen2.5-3B/7B size comparison by language pair: BLEU and term accuracy,
term-list variants, stacked bars (3B base + increment to 7B). See the report
Appendix for the full explanation of the stacking encoding.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "term_expansion" / "by_language_pair" / "shared" / "scripts"))

from by_language_pair_common import (
    SERIES_ORDER,
    STRATEGY_LABELS,
    collect_data,
)
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
    TERM_ACC_YLIM_TOP,
    finalize_pair_layout,
    fixed_top_ylim,
    place_legend_row,
)
from shared.lib.analysis.metrics_loader import LANG_LABELS, LANG_ORDER
from shared.lib.analysis.plot_style import apply_report_style


def _lighten(hex_color: str, factor: float = 0.55) -> str:
    """Blend a hex color toward white by `factor` (0 = unchanged, 1 = white)."""
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    r, g, b = (round(c + (255 - c) * factor) for c in (r, g, b))
    return f"#{r:02x}{g:02x}{b:02x}"


# Verified (see report Appendix A) that Qwen2.5-7B's score is >= 3B's in every
# cell except one near-tie (cleaned/EN-DE term accuracy, 7B ~0.5pp lower) --
# each bar's bottom segment is therefore 3B's actual value and the top segment
# is a real, non-arbitrary quantity: the increment needed to reach 7B's score.
# The one near-tie is handled generically via max(0, ...) (renders as a bare
# 3B-only bar for that one cell) and explained in the appendix prose, not
# special-cased here.
def build_qwen_size_stacked_figure(results_root: Path) -> Figure:
    apply_report_style()
    data = collect_data(results_root)
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
