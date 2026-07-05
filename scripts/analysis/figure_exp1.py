"""Proper terms expansion: original / GPT-expanded / domain-filtered."""

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
    BASELINE_DIRS,
    BASELINE_LABELS,
    DEFAULT_MODE,
    load_metrics_path,
    macro_average,
    require_paths,
)
from plot_style import apply_poster_style

VARIANT_ORDER = ("original", "expand", "cleaned")
VARIANT_LABELS = {
    "original": "Original",
    "expand": "GPT expand",
    "cleaned": "GPT cleaned",
}

TITLE = (
    "Proper Terms Expansion: Original vs GPT Expand vs GPT cleaned\n"
    "(GPT-4o-mini, Qwen2.5-3B, Qwen2.5-7B; proper_term; macro avg over language pairs)"
)


def _collect_data(results_root: Path) -> tuple[dict, dict[str, int]]:
    summaries: dict[str, dict[str, dict]] = {}
    term_counts: dict[str, int] = {}

    paths = [
        results_root / "dev_v1" / variant / baseline / "metrics_summary.json"
        for variant in VARIANT_ORDER
        for baseline in BASELINE_DIRS
    ]
    require_paths(paths)

    for variant in VARIANT_ORDER:
        summaries[variant] = {}
        for baseline in BASELINE_DIRS:
            path = results_root / "dev_v1" / variant / baseline / "metrics_summary.json"
            summary = load_metrics_path(path)
            summaries[variant][baseline] = macro_average(summary, DEFAULT_MODE)
            if baseline == "gpt":
                total = summaries[variant][baseline].get("total_terms")
                if total is not None:
                    term_counts[variant] = int(total)

    return summaries, term_counts


def _variant_legend_label(variant: str, term_counts: dict[str, int]) -> str:
    base = VARIANT_LABELS[variant]
    count = term_counts.get(variant)
    return f"{base} ({count} terms)" if count is not None else base


def build_exp1_figure(results_root: Path) -> Figure:
    apply_poster_style()
    summaries, term_counts = _collect_data(results_root)

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
            group_keys=BASELINE_DIRS,
            group_labels=BASELINE_LABELS,
            series_keys=VARIANT_ORDER,
            series_labels={
                k: _variant_legend_label(k, term_counts) for k in VARIANT_ORDER
            },
            value_fn=lambda variant, baseline: summaries[variant][baseline].get(metric_key),
            ylabel=ylabel,
            xlabel="Model",
            ylim_top=ylim_top,
            yticks=yticks,
        )

    legend_handles = [
        Patch(
            facecolor=EXPANSION_COLORS[v],
            edgecolor="#333333",
            label=_variant_legend_label(v, term_counts),
        )
        for v in VARIANT_ORDER
    ]
    place_side_legend(legend_ax, legend_handles, "Expansion strategy")
    finalize_pair_layout(fig)
    return fig
