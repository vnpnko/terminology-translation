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

NO_TERM_KEY = "no_term"
NO_TERM_MODE = "no_term"
NO_TERM_SOURCE_VARIANT = "original"
NO_TERM_LABEL = "No term"

VARIANT_ORDER = ("original", "expand", "cleaned")
SERIES_ORDER = (NO_TERM_KEY, *VARIANT_ORDER)
VARIANT_LABELS = {
    "original": "Original",
    "expand": "GPT expand",
    "cleaned": "GPT cleaned",
}

TITLE = (
    "Terminology expansion across models\n"
    "(No-term baseline and three proper-term strategies; macro avg over language pairs)"
)


def _collect_data(results_root: Path) -> tuple[dict, dict[str, dict], dict[str, int]]:
    summaries: dict[str, dict[str, dict]] = {}
    no_term: dict[str, dict] = {}
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
            if variant == NO_TERM_SOURCE_VARIANT:
                no_term[baseline] = macro_average(summary, NO_TERM_MODE)
            if baseline == "gpt":
                total = summaries[variant][baseline].get("total_terms")
                if total is not None:
                    term_counts[variant] = int(total)

    return summaries, no_term, term_counts


def _series_legend_label(series: str, term_counts: dict[str, int]) -> str:
    if series == NO_TERM_KEY:
        return NO_TERM_LABEL
    base = VARIANT_LABELS[series]
    count = term_counts.get(series)
    return f"{base} ({count} terms)" if count is not None else base


def build_exp1_figure(results_root: Path) -> Figure:
    apply_poster_style()
    summaries, no_term, term_counts = _collect_data(results_root)

    def value_for(series: str, baseline: str, metric_key: str) -> float | None:
        source = no_term[baseline] if series == NO_TERM_KEY else summaries[series][baseline]
        return source.get(metric_key)

    fig, axes, legend_ax = create_pair_figure(TITLE)

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
            series_keys=SERIES_ORDER,
            series_labels={
                k: _series_legend_label(k, term_counts) for k in SERIES_ORDER
            },
            value_fn=lambda series, baseline, mk=metric_key: value_for(series, baseline, mk),
            ylabel=ylabel,
            xlabel="Model",
            ylim_top=ylim_top,
            yticks=yticks,
        )

    legend_handles = [
        Patch(
            facecolor=EXPANSION_COLORS[v],
            edgecolor="#333333",
            label=_series_legend_label(v, term_counts),
        )
        for v in SERIES_ORDER
    ]
    place_side_legend(legend_ax, legend_handles, "Expansion strategy")
    finalize_pair_layout(fig)
    return fig
