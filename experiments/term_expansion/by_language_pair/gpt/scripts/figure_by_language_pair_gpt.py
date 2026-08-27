"""Terminology expansion strategies by language pair, GPT-4o-mini only."""

from __future__ import annotations

import sys
from pathlib import Path

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
    REPORT_STACK_FIG_SIZE,
    REPORT_STACK_HSPACE,
    REPORT_SUPTITLE_FONTSIZE,
    REPORT_VALUE_FONTSIZE,
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_stack_layout,
    place_legend_row,
    plot_grouped_bars,
)
from shared.lib.analysis.metrics_loader import LANG_LABELS, LANG_ORDER
from shared.lib.analysis.plot_style import apply_report_style


def build_by_language_pair_gpt_figure(results_root: Path) -> Figure:
    apply_report_style()
    data = collect_data(results_root)["gpt"]

    fig, axes, _ = create_pair_figure(
        None,
        figsize=REPORT_STACK_FIG_SIZE,
        hspace=REPORT_STACK_HSPACE,
        legend_position="bottom",
        stacked=True,
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

    axes[0].tick_params(labelbottom=False)

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
    finalize_stack_layout(fig)
    return fig
