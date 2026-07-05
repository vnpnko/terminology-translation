"""Term expansion: original vs GPT contextual vs domain-filtered."""

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

STRATEGY_ORDER = ("original", "expand", "cleaned")
STRATEGY_LABELS = {
    "original": "Original",
    "expand": "GPT expand",
    "cleaned": "GPT cleaned",
}

TITLE = (
    "Proper Terms Expansion: Original vs GPT Expand vs GPT cleaned\n"
    "(GPT-4o-mini; proper_term; dev_v1)"
)

def _collect_data(results_root: Path) -> dict[str, dict[str, dict[str, float | None]]]:
    paths = [
        results_root / "dev_v1" / strategy / "gpt" / "metrics_summary.json"
        for strategy in STRATEGY_ORDER
    ]
    require_paths(paths)

    data: dict[str, dict[str, dict[str, float | None]]] = {}
    for strategy in STRATEGY_ORDER:
        summary = load_metrics_path(
            results_root / "dev_v1" / strategy / "gpt" / "metrics_summary.json"
        )
        data[strategy] = {
            lang: get_lang_mode_metrics(summary, lang, DEFAULT_MODE) for lang in LANG_ORDER
        }
    return data


def build_exp23_figure(results_root: Path) -> Figure:
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
            series_keys=STRATEGY_ORDER,
            series_labels=STRATEGY_LABELS,
            value_fn=lambda strategy, lang: data[strategy][lang].get(metric_key),
            ylabel=ylabel,
            xlabel="Language pair",
            ylim_top=ylim_top,
            yticks=yticks,
        )

    legend_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=STRATEGY_LABELS[k])
        for k in STRATEGY_ORDER
    ]
    place_side_legend(legend_ax, legend_handles, "Expansion strategy")
    finalize_pair_layout(fig)
    return fig
