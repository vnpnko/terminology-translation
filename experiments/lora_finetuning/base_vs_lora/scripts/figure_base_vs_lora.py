"""Qwen base (few-shot) vs LoRA 1 epoch (zero-shot): one chart per model size, 5 metric groups."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "scripts"))

from shared.lib.analysis.grouped_bar_figure_common import (
    EXPANSION_COLORS,
    REPORT_LEGEND_ROW_FONTSIZE,
    REPORT_LEGEND_ROW_NCOL,
    REPORT_STACK_FIG_SIZE,
    REPORT_STACK_HSPACE,
    REPORT_SUPTITLE_FONTSIZE,
    create_pair_figure,
    finalize_stack_layout,
    place_legend_row,
    plot_grouped_bars,
)
from shared.lib.analysis.plot_style import apply_report_style
from compare_common import (  # noqa: E402
    RUN_LORA_1_EPOCH_ZERO_SHOT,
    extract_row,
    load_metrics_summary,
    load_registry,
)

FINETUNING_RESULTS = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "results"
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "run_registry.json"

LANG_ORDER = ("ende", "enes", "enru")
MODEL_KEYS = ("3B", "7B")
MODEL_LABELS = {"3B": "Qwen 3B", "7B": "Qwen 7B"}

BASE_KEY = "qwen_base_few_shot"
LORA_KEY = "qwen_lora_zero_shot"
SERIES_ORDER = (BASE_KEY, LORA_KEY)
SERIES_LABELS = {
    BASE_KEY: "Base (few-shot)",
    LORA_KEY: "LoRA (zero-shot, 1 epoch)",
}

# Consistency metrics come back as 0-1 ratios; scale to percent for a shared 0-100 axis
# with BLEU/chrF/Term Accuracy — display-only, the underlying data/xlsx are untouched.
PERCENT_SCALE_METRICS = ("macro_avg_consistency", "weighted_avg_consistency")

METRIC_ORDER = ("bleu", "chrf", "term_accuracy_pct", "macro_avg_consistency", "weighted_avg_consistency")
METRIC_LABELS = {
    "bleu": "BLEU",
    "chrf": "chrF",
    "term_accuracy_pct": "Term\nAcc (%)",
    "macro_avg_consistency": "Macro\nCons (%)",
    "weighted_avg_consistency": "Weighted\nCons (%)",
}

YLIM_TOP = 100


def lora_1_epoch_zero_shot_run(registry: dict[str, Any], model_key: str) -> dict[str, Any]:
    runs = {run["run_id"]: run for run in registry[model_key]["runs"]}
    return runs[RUN_LORA_1_EPOCH_ZERO_SHOT]


def macro_row(rows: dict[str, list[float | None]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for i, metric in enumerate(METRIC_ORDER):
        vals = [rows[lang][i] for lang in LANG_ORDER if rows[lang][i] is not None]
        val = round(sum(vals) / len(vals), 2) if vals else None
        if metric in PERCENT_SCALE_METRICS and val is not None:
            val *= 100
        out[metric] = val
    return out


def _collect_data() -> dict[str, dict[str, dict[str, float | None]]]:
    registry = load_registry(REGISTRY_PATH)
    data: dict[str, dict[str, dict[str, float | None]]] = {}
    for model_key in MODEL_KEYS:
        model_dir = registry[model_key]["model_dir"]
        base_summary = load_metrics_summary(FINETUNING_RESULTS / model_dir / "qwen_base" / "metrics_summary.json")
        lora_run = lora_1_epoch_zero_shot_run(registry, model_key)
        lora_summary = load_metrics_summary(FINETUNING_RESULTS / model_dir / lora_run["folder"] / "metrics_summary.json")
        data[model_key] = {
            BASE_KEY: macro_row({lang: extract_row(base_summary, lang) for lang in LANG_ORDER}),
            LORA_KEY: macro_row({lang: extract_row(lora_summary, lang) for lang in LANG_ORDER}),
        }
    return data


def build_base_vs_lora_figure(project_root: Path) -> Figure:
    apply_report_style()
    data = _collect_data()

    fig, axes, _ = create_pair_figure(
        None,
        figsize=REPORT_STACK_FIG_SIZE,
        hspace=REPORT_STACK_HSPACE,
        legend_position="bottom",
        stacked=True,
    )

    for ax, model_key in zip(axes, MODEL_KEYS):
        model_data = data[model_key]
        plot_grouped_bars(
            ax,
            group_keys=METRIC_ORDER,
            group_labels=METRIC_LABELS,
            series_keys=SERIES_ORDER,
            series_labels=SERIES_LABELS,
            value_fn=lambda series, metric, md=model_data: md[series][metric],
            ylabel="Score",
            panel_title=MODEL_LABELS[model_key],
            panel_title_fontsize=REPORT_SUPTITLE_FONTSIZE,
            ylim_top=YLIM_TOP,
            yticks=list(range(0, YLIM_TOP + 1, 20)),
            show_values=False,
        )

    axes[0].tick_params(labelbottom=False)

    legend_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=SERIES_LABELS[k])
        for k in SERIES_ORDER
    ]
    place_legend_row(
        fig,
        legend_handles,
        fontsize=REPORT_LEGEND_ROW_FONTSIZE,
        ncol=REPORT_LEGEND_ROW_NCOL,
    )
    finalize_stack_layout(fig)
    return fig
