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
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_pair_layout,
    place_side_legend,
    plot_grouped_bars,
)
from shared.lib.analysis.metrics_loader import (
    BASELINE_DIRS,
    BASELINE_LABELS,
    DEFAULT_MODE,
    EXTERNAL_DICTIONARY_RESULTS,
    NO_TERM_MODE,
    RANDOM_TERM_MODE,
    load_metrics_path,
    macro_average,
    require_paths,
)
from shared.lib.analysis.plot_style import apply_poster_style

NO_TERM_KEY = "no_term"
RANDOM_TERM_KEY = "random_term"
EXTERNAL_DICTIONARY_KEY = "external_dictionary"
BASELINE_SOURCE_VARIANT = "original"
NO_TERM_LABEL = "No term"
RANDOM_TERM_LABEL = "Random term"
EXTERNAL_DICTIONARY_LABEL = "External dictionary"

VARIANT_ORDER = ("original", "expand", "cleaned")
SERIES_ORDER = (NO_TERM_KEY, RANDOM_TERM_KEY, *VARIANT_ORDER, EXTERNAL_DICTIONARY_KEY)
VARIANT_LABELS = {
    "original": "Original",
    "expand": "GPT-expanded",
    "cleaned": "GPT-cleaned",
}

TITLE = (
    "Terminology expansion across models\n"
    "(dev_v1; macro avg over language pairs)"
)


def _variant_dir(results_root: Path, variant: str) -> Path:
    if variant == BASELINE_SOURCE_VARIANT:
        return results_root / "dev_v1" / variant / "few_shot"
    return results_root / "dev_v1" / variant


def _collect_data(
    results_root: Path,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, int],
]:
    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    no_term: dict[str, dict[str, Any]] = {}
    random_term: dict[str, dict[str, Any]] = {}
    external_dictionary: dict[str, dict[str, Any]] = {}
    term_counts: dict[str, int] = {}

    paths = [
        _variant_dir(results_root, variant) / baseline / "metrics_summary.json"
        for variant in VARIANT_ORDER
        for baseline in BASELINE_DIRS
    ]
    paths.extend(
        results_root / EXTERNAL_DICTIONARY_RESULTS / baseline / "metrics_summary.json"
        for baseline in BASELINE_DIRS
    )
    require_paths(paths)

    for variant in VARIANT_ORDER:
        summaries[variant] = {}
        for baseline in BASELINE_DIRS:
            path = _variant_dir(results_root, variant) / baseline / "metrics_summary.json"
            summary = load_metrics_path(path)
            summaries[variant][baseline] = macro_average(summary, DEFAULT_MODE)
            if variant == BASELINE_SOURCE_VARIANT:
                no_term[baseline] = macro_average(summary, NO_TERM_MODE)
                random_term[baseline] = macro_average(summary, RANDOM_TERM_MODE)
            if baseline == "gpt":
                total = summaries[variant][baseline].get("total_terms")
                if total is not None:
                    term_counts[variant] = int(total)

    for baseline in BASELINE_DIRS:
        path = results_root / EXTERNAL_DICTIONARY_RESULTS / baseline / "metrics_summary.json"
        summary = load_metrics_path(path)
        external_dictionary[baseline] = macro_average(summary, DEFAULT_MODE)

    gpt_total = external_dictionary["gpt"].get("total_terms")
    if gpt_total is not None:
        term_counts[EXTERNAL_DICTIONARY_KEY] = int(gpt_total)

    return summaries, no_term, random_term, external_dictionary, term_counts


def _series_legend_label(series: str, term_counts: dict[str, int]) -> str:
    if series == NO_TERM_KEY:
        return NO_TERM_LABEL
    if series == RANDOM_TERM_KEY:
        return RANDOM_TERM_LABEL
    if series == EXTERNAL_DICTIONARY_KEY:
        count = term_counts.get(series)
        return (
            f"{EXTERNAL_DICTIONARY_LABEL} ({count} terms)"
            if count is not None
            else EXTERNAL_DICTIONARY_LABEL
        )
    base = VARIANT_LABELS[series]
    count = term_counts.get(series)
    return f"{base} ({count} terms)" if count is not None else base


def build_by_model_figure(results_root: Path) -> Figure:
    apply_poster_style()
    summaries, no_term, random_term, external_dictionary, term_counts = _collect_data(results_root)

    def value_for(series: str, baseline: str, metric_key: str) -> float | None:
        if series == NO_TERM_KEY:
            source = no_term[baseline]
        elif series == RANDOM_TERM_KEY:
            source = random_term[baseline]
        elif series == EXTERNAL_DICTIONARY_KEY:
            source = external_dictionary[baseline]
        else:
            source = summaries[series][baseline]
        return source.get(metric_key)

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
