"""Proper terms expansion: original / GPT-expanded / domain-filtered."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

from figure_common import (
    EXPANSION_COLORS,
    create_pair_figure,
    finalize_pair_layout,
    place_figure_caption,
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
    "cleaned": "Domain-filtered",
}

TITLE = (
    "Proper Terms Expansion: Original vs GPT Expand vs Domain-Filtered\n"
    "(GPT-4o-mini, Qwen2.5-3B, Qwen2.5-7B · proper_term · macro avg over EN→DE/RU/ES)"
)
SUBTITLE = (
    "Same original/expand lists as Term Expansion Strategies; third arm is domain-filtered "
    "(not dictionary). Term counts (GPT) in legend."
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

    for ax, metric_key, ylabel, panel_title in [
        (axes[0], "bleu", "BLEU", "BLEU by model"),
        (axes[1], "term_accuracy_pct", "Term accuracy (%)", "Term accuracy by model"),
    ]:
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
            panel_title=panel_title,
            xlabel="Model",
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
    place_figure_caption(fig, SUBTITLE)
    finalize_pair_layout(fig)
    return fig
