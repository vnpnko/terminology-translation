"""dev_v1 vs dev_v2 terminology dataset comparison (GPT/Qwen 3B/Qwen 7B), one figure per term mode."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
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
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_pair_layout,
    place_legend_row,
    plot_grouped_bars,
)
from shared.lib.analysis.metrics_loader import (
    BASELINE_DIRS,
    BASELINE_LABELS,
    EXTERNAL_DICTIONARY_RESULTS,
    MODE_ORDER,
    load_metrics_path,
    macro_average,
    require_paths,
)
from shared.lib.analysis.plot_style import apply_report_style

DATASET_ORDER = ("dev_v1", "dev_v2")
DATASET_LABELS = {
    "dev_v1": "dev_v1",
    "dev_v2": "dev_v2",
}
DATASET_COLORS = {
    "dev_v1": EXPANSION_COLORS["dev_v1"],
    "dev_v2": EXPANSION_COLORS["dev_v2"],
}


def _dataset_dir(results_root: Path, dataset: str) -> Path:
    if dataset == "dev_v1":
        return results_root / "dev_v1" / "original" / "few_shot"
    return results_root / EXTERNAL_DICTIONARY_RESULTS


def _collect_data(results_root: Path, mode: str) -> dict[str, dict[str, dict[str, float | None]]]:
    paths = [
        _dataset_dir(results_root, dataset) / baseline / "metrics_summary.json"
        for dataset in DATASET_ORDER
        for baseline in BASELINE_DIRS
    ]
    require_paths(paths)

    data: dict[str, dict[str, dict[str, float | None]]] = {}
    for dataset in DATASET_ORDER:
        data[dataset] = {}
        for baseline in BASELINE_DIRS:
            path = _dataset_dir(results_root, dataset) / baseline / "metrics_summary.json"
            summary = load_metrics_path(path)
            data[dataset][baseline] = macro_average(summary, mode)
    return data


def build_dataset_comparison_figure(results_root: Path, mode: str) -> Figure:
    apply_report_style()
    data = _collect_data(results_root, mode)

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
            group_keys=BASELINE_DIRS,
            group_labels=BASELINE_LABELS,
            series_keys=DATASET_ORDER,
            series_labels=DATASET_LABELS,
            value_fn=lambda dataset, baseline, mk=metric_key: data[dataset][baseline].get(mk),
            ylabel=ylabel,
            ylim_top=ylim_top,
            yticks=yticks,
            show_values=False,
            group_width=REPORT_GROUP_WIDTH,
            bar_width_ratio=REPORT_BAR_WIDTH_RATIO,
        )

    legend_handles = [
        Patch(facecolor=DATASET_COLORS[k], edgecolor="#333333", label=DATASET_LABELS[k])
        for k in DATASET_ORDER
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


def build_dataset_comparison_figures(results_root: Path) -> dict[str, Figure]:
    return {mode: build_dataset_comparison_figure(results_root, mode) for mode in MODE_ORDER}
