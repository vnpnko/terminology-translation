"""dev_v1 (test) vs dev_v2 training set comparison (GPT baseline)."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

from figure_common import (
    BLEU_YLIM_TOP,
    EXPANSION_COLORS,
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_pair_layout,
    place_side_legend,
    plot_grouped_bars,
)
from metrics_loader import (
    DEFAULT_MODE,
    LANG_LABELS,
    LANG_ORDER,
    get_lang_mode_metrics,
    load_metrics_path,
    require_paths,
)
from plot_style import apply_poster_style

DATASET_ORDER = ("dev_v1", "dev_v2_training")
DATASET_LABELS = {
    "dev_v1": "Test (dev_v1)",
    "dev_v2_training": "Training (dev_v2)",
}
DATASET_COLORS = {
    "dev_v1": EXPANSION_COLORS["dev_v1"],
    "dev_v2_training": EXPANSION_COLORS["dev_v2_training"],
}

TITLE = (
    "Test vs training terminology datasets\n"
    "(proper_term; GPT-4o-mini; per language pair)"
)


def _dataset_paths(results_root: Path) -> dict[str, Path]:
    return {
        "dev_v1": results_root / "dev_v1" / "original" / "gpt" / "metrics_summary.json",
        "dev_v2_training": results_root / "dev_v2" / "training" / "gpt" / "metrics_summary.json",
    }


def _collect_data(results_root: Path) -> dict[str, dict[str, dict[str, float | None]]]:
    paths = _dataset_paths(results_root)
    require_paths(list(paths.values()))

    data: dict[str, dict[str, dict[str, float | None]]] = {}
    for key, path in paths.items():
        summary = load_metrics_path(path)
        data[key] = {
            lang: get_lang_mode_metrics(summary, lang, DEFAULT_MODE) for lang in LANG_ORDER
        }
    return data


def build_exp4_figure(results_root: Path) -> Figure:
    apply_poster_style()
    data = _collect_data(results_root)

    fig, axes, legend_ax = create_pair_figure(TITLE)

    metric_axes = [
        (axes[0], "bleu", "BLEU", BLEU_YLIM_TOP, list(range(0, BLEU_YLIM_TOP + 1, 10))),
        (
            axes[1],
            "term_accuracy_pct",
            "Term accuracy (%)",
            TERM_ACC_YLIM_TOP,
            list(range(0, TERM_ACC_YLIM_TOP + 1, 20)),
        ),
    ]
    for ax, metric_key, ylabel, ylim_top, yticks in metric_axes:
        plot_grouped_bars(
            ax,
            group_keys=LANG_ORDER,
            group_labels=LANG_LABELS,
            series_keys=DATASET_ORDER,
            series_labels=DATASET_LABELS,
            value_fn=lambda dataset, lang, mk=metric_key: data[dataset][lang].get(mk),
            ylabel=ylabel,
            xlabel="Language pair",
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
