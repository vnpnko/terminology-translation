"""Zero-shot vs few-shot for GPT-4o-mini: one chart, 5 metric groups, macro-averaged over languages."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.lib.analysis.grouped_bar_figure_common import plot_grouped_bars
from shared.lib.analysis.metrics_loader import load_summary, macro_average
from shared.lib.analysis.plot_style import POSTER_BLUE, apply_poster_style

ZERO_SHOT_SUMMARY = PROJECT_ROOT / "shared" / "results" / "dev_v1" / "original" / "zero_shot" / "gpt" / "metrics_summary.json"
FEW_SHOT_SUMMARY = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "results" / "gpt_base" / "metrics_summary.json"

# Consistency metrics come back as 0-1 ratios; scale to percent for a shared 0-100 axis
# with BLEU/chrF/Term Accuracy — display-only, the underlying data/xlsx are untouched.
PERCENT_SCALE_METRICS = ("macro_avg_consistency", "weighted_avg_consistency")

METRIC_ORDER = ("bleu", "chrf", "term_accuracy_pct", "macro_avg_consistency", "weighted_avg_consistency")
METRIC_LABELS = {
    "bleu": "BLEU",
    "chrf": "chrF",
    "term_accuracy_pct": "Term Accuracy %",
    "macro_avg_consistency": "Macro\nConsistency (%)",
    "weighted_avg_consistency": "Weighted\nConsistency (%)",
}

ZERO_SHOT_KEY = "zero_shot"
FEW_SHOT_KEY = "few_shot"
SERIES_ORDER = (ZERO_SHOT_KEY, FEW_SHOT_KEY)
SERIES_LABELS = {
    ZERO_SHOT_KEY: "Zero-shot",
    FEW_SHOT_KEY: "Few-shot",
}

TITLE = "GPT-4o-mini: zero-shot vs. few-shot\n(train dev_v2; test dev_v1; proper_term)"

FIG_SIZE = (13, 6.5)
YLIM_TOP = 100


def _collect_data() -> dict[str, dict[str, float | None]]:
    data: dict[str, dict[str, float | None]] = {}
    for series_key, path in ((ZERO_SHOT_KEY, ZERO_SHOT_SUMMARY), (FEW_SHOT_KEY, FEW_SHOT_SUMMARY)):
        summary = load_summary(path)
        row = macro_average(summary, "proper_term")
        for metric in PERCENT_SCALE_METRICS:
            if row.get(metric) is not None:
                row[metric] = row[metric] * 100
        data[series_key] = row
    return data


def build_few_shot_ablation_figure(project_root: Path) -> Figure:
    apply_poster_style()
    data = _collect_data()

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    fig.suptitle(TITLE, fontsize=16, fontweight="bold", color=POSTER_BLUE, y=0.98)

    plot_grouped_bars(
        ax,
        group_keys=METRIC_ORDER,
        group_labels=METRIC_LABELS,
        series_keys=SERIES_ORDER,
        series_labels=SERIES_LABELS,
        value_fn=lambda series, metric: data[series][metric],
        ylabel="Score",
        ylim_top=YLIM_TOP,
        yticks=list(range(0, YLIM_TOP + 1, 10)),
    )

    ax.legend(title="Prompting", loc="upper right", framealpha=0.95)
    fig.subplots_adjust(top=0.84, bottom=0.12, left=0.08, right=0.97)
    return fig
