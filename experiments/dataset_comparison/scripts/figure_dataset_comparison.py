"""dev_v1 vs dev_v2 terminology dataset comparison (GPT/Qwen 3B/Qwen 7B), one figure per term mode."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.figure_common import (
    BLEU_YLIM_TOP,
    EXPANSION_COLORS,
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_pair_layout,
    place_side_legend,
    plot_grouped_bars,
)
from src.analysis.metrics_loader import (
    BASELINE_DIRS,
    BASELINE_LABELS,
    EXTERNAL_DICTIONARY_RESULTS,
    MODE_ORDER,
    load_metrics_path,
    macro_average,
    require_paths,
)
from src.analysis.plot_style import apply_poster_style

DATASET_ORDER = ("dev_v1", "dev_v2")
DATASET_LABELS = {
    "dev_v1": "dev_v1",
    "dev_v2": "dev_v2",
}
DATASET_COLORS = {
    "dev_v1": EXPANSION_COLORS["dev_v1"],
    "dev_v2": EXPANSION_COLORS["dev_v2"],
}


def _title(mode: str) -> str:
    return (
        "dev_v1 vs dev_v2 terminology datasets\n"
        f"({mode}; macro avg over language pairs; per model)"
    )


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
    apply_poster_style()
    data = _collect_data(results_root, mode)

    fig, axes, legend_ax = create_pair_figure(_title(mode))

    metric_axes = [
        (axes[0], "bleu", "BLEU (macro avg)", BLEU_YLIM_TOP, list(range(0, BLEU_YLIM_TOP + 1, 10))),
        (
            axes[1],
            "term_accuracy_pct",
            "Term accuracy (%) (macro avg)",
            TERM_ACC_YLIM_TOP,
            list(range(0, TERM_ACC_YLIM_TOP + 1, 20)),
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
            xlabel="Model",
            ylim_top=ylim_top,
            yticks=yticks,
        )

    legend_handles = [
        Patch(facecolor=DATASET_COLORS[k], edgecolor="#333333", label=DATASET_LABELS[k])
        for k in DATASET_ORDER
    ]
    place_side_legend(legend_ax, legend_handles, "Dataset")
    finalize_pair_layout(fig)
    return fig


def build_dataset_comparison_figures(results_root: Path) -> dict[str, Figure]:
    return {mode: build_dataset_comparison_figure(results_root, mode) for mode in MODE_ORDER}
