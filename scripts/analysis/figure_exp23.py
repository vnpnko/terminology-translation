"""Term expansion: original vs GPT contextual vs dictionary from dev_v2."""

from __future__ import annotations

from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

from figure_common import (
    EXPANSION_COLORS,
    compute_shared_expansion_ylims,
    create_pair_figure,
    finalize_pair_layout,
    place_figure_caption,
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

STRATEGY_ORDER = ("original", "expand", "dictionary")
STRATEGY_LABELS = {
    "original": "Original",
    "expand": "GPT expand",
    "dictionary": "Dictionary",
}

TITLE = (
    "Term Expansion Strategies: Original vs GPT Expand vs Dictionary from dev_v2\n"
    "(GPT-4o-mini · proper_term · 500-line dev_v1 · by language pair)"
)
SUBTITLE = (
    "Same original/expand lists as Proper Terms Expansion; third arm is dictionary from dev_v2. "
    "EN→ES scores highest; GPT expand leads on BLEU/chrF, dictionary matches expand on BLEU."
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
    shared_ylims = compute_shared_expansion_ylims(results_root)

    fig, axes, legend_ax = create_pair_figure(TITLE)

    for ax, metric_key, ylabel, panel_title in [
        (axes[0], "bleu", "BLEU", "BLEU by language pair"),
        (axes[1], "term_accuracy_pct", "Term accuracy (%)", "Term accuracy by language pair"),
    ]:
        plot_grouped_bars(
            ax,
            group_keys=LANG_ORDER,
            group_labels=LANG_LABELS,
            series_keys=STRATEGY_ORDER,
            series_labels=STRATEGY_LABELS,
            value_fn=lambda strategy, lang: data[strategy][lang].get(metric_key),
            ylabel=ylabel,
            panel_title=panel_title,
            xlabel="Language pair",
            ylim=shared_ylims[metric_key],
        )

    legend_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=STRATEGY_LABELS[k])
        for k in STRATEGY_ORDER
    ]
    place_side_legend(legend_ax, legend_handles, "Expansion strategy")
    place_figure_caption(fig, SUBTITLE)
    finalize_pair_layout(fig)
    return fig
