"""Terminology expansion strategies aggregated by model (macro avg over language pairs)."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

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
    REPORT_STACK_FIG_SIZE,
    REPORT_STACK_HSPACE,
    REPORT_SUPTITLE_FONTSIZE,
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_stack_layout,
    place_legend_row,
    plot_grouped_bars,
)
from shared.lib.analysis.metrics_loader import (
    BASELINE_DIRS,
    BASELINE_LABELS,
    DEFAULT_MODE,
    load_metrics_path,
    macro_average,
    require_paths,
)
from shared.lib.analysis.plot_style import apply_report_style

BASELINE_SOURCE_VARIANT = "original"

# "dictionary" reads shared/results/dev_v1/dictionary/ (the real term-list
# variant built by experiments/term_expansion/dictionary/'s
# build_term_dictionary.py/apply_dictionary_to_dev_v1.py) via the same
# _variant_dir() fallback used for expand/cleaned below -- not the unrelated
# dev_v2-on-its-own-terms comparison the old EXTERNAL_DICTIONARY_* constants
# pointed at (that data is shared/results/dev_v2/, i.e. the dataset_comparison
# experiment's dev_v2 row, unrelated to this term-list-variant axis).
VARIANT_ORDER = ("original", "expand", "cleaned", "dictionary")
SERIES_ORDER = VARIANT_ORDER
VARIANT_LABELS = {
    "original": "Original",
    "expand": "GPT-expanded",
    "cleaned": "GPT-cleaned",
    "dictionary": "Dictionary",
}

def _variant_dir(results_root: Path, variant: str) -> Path:
    if variant == BASELINE_SOURCE_VARIANT:
        return results_root / "dev_v1" / variant / "few_shot"
    return results_root / "dev_v1" / variant


def _collect_data(
    results_root: Path,
) -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    term_counts: dict[str, int] = {}

    paths = [
        _variant_dir(results_root, variant) / baseline / "metrics_summary.json"
        for variant in VARIANT_ORDER
        for baseline in BASELINE_DIRS
    ]
    require_paths(paths)

    for variant in VARIANT_ORDER:
        summaries[variant] = {}
        for baseline in BASELINE_DIRS:
            path = _variant_dir(results_root, variant) / baseline / "metrics_summary.json"
            summary = load_metrics_path(path)
            summaries[variant][baseline] = macro_average(summary, DEFAULT_MODE)
            if baseline == "gpt":
                total = summaries[variant][baseline].get("total_terms")
                if total is not None:
                    term_counts[variant] = int(total)

    return summaries, term_counts


def _series_legend_label(series: str, term_counts: dict[str, int]) -> str:
    base = VARIANT_LABELS[series]
    count = term_counts.get(series)
    return f"{base} ({count} terms)" if count is not None else base


def build_by_model_figure(results_root: Path) -> Figure:
    apply_report_style()
    summaries, term_counts = _collect_data(results_root)

    def value_for(series: str, baseline: str, metric_key: str) -> float | None:
        return summaries[series][baseline].get(metric_key)

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
            group_keys=BASELINE_DIRS,
            group_labels=BASELINE_LABELS,
            series_keys=SERIES_ORDER,
            series_labels={
                k: _series_legend_label(k, term_counts) for k in SERIES_ORDER
            },
            value_fn=lambda series, baseline, mk=metric_key: value_for(series, baseline, mk),
            ylabel=ylabel,
            ylim_top=ylim_top,
            yticks=yticks,
            panel_title_fontsize=REPORT_SUPTITLE_FONTSIZE,
            show_values=False,
            group_width=REPORT_GROUP_WIDTH,
            bar_width_ratio=REPORT_BAR_WIDTH_RATIO,
        )

    axes[0].tick_params(labelbottom=False)

    legend_handles = [
        Patch(
            facecolor=EXPANSION_COLORS[v],
            edgecolor="#333333",
            label=_series_legend_label(v, term_counts),
        )
        for v in SERIES_ORDER
    ]
    place_legend_row(
        fig,
        legend_handles,
        fontsize=REPORT_LEGEND_ROW_FONTSIZE,
        ncol=2,
    )
    finalize_stack_layout(fig)
    return fig
