"""Terminology expansion strategies by language pair, one figure per model (GPT/Qwen 3B/Qwen 7B)."""

from __future__ import annotations

import sys
from pathlib import Path

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
