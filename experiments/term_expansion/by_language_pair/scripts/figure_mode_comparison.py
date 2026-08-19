"""Term expansion: original vs GPT contextual vs domain-filtered."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
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
    DEFAULT_MODE,
    EXTERNAL_DICTIONARY_RESULTS,
    LANG_LABELS,
    LANG_ORDER,
    NO_TERM_MODE,
    RANDOM_TERM_MODE,
    get_lang_mode_metrics,
    load_metrics_path,
    require_paths,
)
from src.analysis.plot_style import apply_poster_style

NO_TERM_KEY = "no_term"
RANDOM_TERM_KEY = "random_term"
EXTERNAL_DICTIONARY_KEY = "external_dictionary"
BASELINE_SOURCE_STRATEGY = "original"
NO_TERM_LABEL = "No term"
RANDOM_TERM_LABEL = "Random term"
EXTERNAL_DICTIONARY_LABEL = "External dictionary"

STRATEGY_ORDER = ("original", "expand", "cleaned")
SERIES_ORDER = (NO_TERM_KEY, RANDOM_TERM_KEY, *STRATEGY_ORDER, EXTERNAL_DICTIONARY_KEY)
STRATEGY_LABELS = {
    NO_TERM_KEY: NO_TERM_LABEL,
    RANDOM_TERM_KEY: RANDOM_TERM_LABEL,
    "original": "Original",
    "expand": "GPT expand",
    "cleaned": "GPT cleaned",
    EXTERNAL_DICTIONARY_KEY: EXTERNAL_DICTIONARY_LABEL,
}

TITLE = (
    "Terminology expansion by language pair\n"
    "(dev_v1 baselines/strategies plus External dictionary on dev_v2; evaluated with GPT-4o-mini)"
)

def _strategy_dir(results_root: Path, strategy: str) -> Path:
    if strategy == BASELINE_SOURCE_STRATEGY:
        return results_root / "dev_v1" / strategy / "zero_shot"
    return results_root / "dev_v1" / strategy


def _collect_data(results_root: Path) -> dict[str, dict[str, dict[str, float | None]]]:
    paths = [
        _strategy_dir(results_root, strategy) / "gpt" / "metrics_summary.json"
        for strategy in STRATEGY_ORDER
    ]
    paths.append(
        results_root / EXTERNAL_DICTIONARY_RESULTS / "gpt" / "metrics_summary.json"
    )
    require_paths(paths)

    data: dict[str, dict[str, dict[str, float | None]]] = {}
    for strategy in STRATEGY_ORDER:
        summary = load_metrics_path(
            _strategy_dir(results_root, strategy) / "gpt" / "metrics_summary.json"
        )
        data[strategy] = {
            lang: get_lang_mode_metrics(summary, lang, DEFAULT_MODE) for lang in LANG_ORDER
        }
        if strategy == BASELINE_SOURCE_STRATEGY:
            data[NO_TERM_KEY] = {
                lang: get_lang_mode_metrics(summary, lang, NO_TERM_MODE)
                for lang in LANG_ORDER
            }
            data[RANDOM_TERM_KEY] = {
                lang: get_lang_mode_metrics(summary, lang, RANDOM_TERM_MODE)
                for lang in LANG_ORDER
            }

    external_summary = load_metrics_path(
        results_root / EXTERNAL_DICTIONARY_RESULTS / "gpt" / "metrics_summary.json"
    )
    data[EXTERNAL_DICTIONARY_KEY] = {
        lang: get_lang_mode_metrics(external_summary, lang, DEFAULT_MODE)
        for lang in LANG_ORDER
    }
    return data


def build_mode_comparison_figure(results_root: Path) -> Figure:
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
            series_keys=SERIES_ORDER,
            series_labels=STRATEGY_LABELS,
            value_fn=lambda strategy, lang: data[strategy][lang].get(metric_key),
            ylabel=ylabel,
            xlabel="Language pair",
            ylim_top=ylim_top,
            yticks=yticks,
        )

    legend_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=STRATEGY_LABELS[k])
        for k in SERIES_ORDER
    ]
    place_side_legend(legend_ax, legend_handles, "Expansion strategy")
    finalize_pair_layout(fig)
    return fig
