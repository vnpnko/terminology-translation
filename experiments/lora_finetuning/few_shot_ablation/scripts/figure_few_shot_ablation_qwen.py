"""Zero-shot vs few-shot for Qwen 3B/7B: 2x2 grid (model size x base/LoRA), 5 metric bars per shot-mode group."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "scripts"))

from shared.lib.analysis.grouped_bar_figure_common import EXPANSION_COLORS, plot_grouped_bars
from shared.lib.analysis.plot_style import POSTER_BLUE, apply_poster_style
from compare_common import (  # noqa: E402
    RUN_LORA_1_EPOCH_FEW_SHOT,
    RUN_LORA_1_EPOCH_ZERO_SHOT,
    extract_row,
    load_metrics_summary,
    load_registry,
)

BASELINE_RESULTS_ROOT = PROJECT_ROOT / "shared" / "results"
FINETUNING_RESULTS = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "results"
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "run_registry.json"

LANG_ORDER = ("ende", "enes", "enru")
MODEL_KEYS = ("3B", "7B")
MODEL_LABELS = {"3B": "Qwen 3B", "7B": "Qwen 7B"}
BASELINE_DIRS = {"3B": "qwen_3b", "7B": "qwen_7b"}

BLOCK_BASE = "qwen_base"
BLOCK_LORA = "qwen_lora"
BLOCK_ORDER = (BLOCK_BASE, BLOCK_LORA)
BLOCK_TITLES = {BLOCK_BASE: "Base (untrained)", BLOCK_LORA: "LoRA (1 epoch)"}

ZERO_SHOT_KEY = "zero_shot"
FEW_SHOT_KEY = "few_shot"
SHOT_ORDER = (ZERO_SHOT_KEY, FEW_SHOT_KEY)
SHOT_LABELS = {ZERO_SHOT_KEY: "Zero-shot", FEW_SHOT_KEY: "Few-shot"}

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

TITLE = "Qwen (base & LoRA): zero-shot vs. few-shot\n(train dev_v2; test dev_v1; proper_term)"

FIG_SIZE = (17, 12)
YLIM_TOP = 100


def macro_row(rows: dict[str, list[float | None]]) -> dict[str, float | None]:
    out: dict[str, float | None] = {}
    for i, metric in enumerate(METRIC_ORDER):
        vals = [rows[lang][i] for lang in LANG_ORDER if rows[lang][i] is not None]
        val = round(sum(vals) / len(vals), 2) if vals else None
        if metric in PERCENT_SCALE_METRICS and val is not None:
            val *= 100
        out[metric] = val
    return out


def qwen_base_rows(model_dir: str, baseline_dir: str) -> dict[str, list[float | None]]:
    zero_shot_summary = load_metrics_summary(
        BASELINE_RESULTS_ROOT / "dev_v1" / "original" / "zero_shot" / baseline_dir / "metrics_summary.json"
    )
    few_shot_summary = load_metrics_summary(FINETUNING_RESULTS / model_dir / "qwen_base" / "metrics_summary.json")
    return {
        ZERO_SHOT_KEY: macro_row({lang: extract_row(zero_shot_summary, lang) for lang in LANG_ORDER}),
        FEW_SHOT_KEY: macro_row({lang: extract_row(few_shot_summary, lang) for lang in LANG_ORDER}),
    }


def qwen_lora_rows(registry: dict[str, Any], model_key: str, model_dir: str) -> dict[str, list[float | None]]:
    runs = {run["run_id"]: run for run in registry[model_key]["runs"]}
    zero_shot_run = runs[RUN_LORA_1_EPOCH_ZERO_SHOT]
    few_shot_run = runs[RUN_LORA_1_EPOCH_FEW_SHOT]
    zero_shot_summary = load_metrics_summary(FINETUNING_RESULTS / model_dir / zero_shot_run["folder"] / "metrics_summary.json")
    few_shot_summary = load_metrics_summary(FINETUNING_RESULTS / model_dir / few_shot_run["folder"] / "metrics_summary.json")
    return {
        ZERO_SHOT_KEY: macro_row({lang: extract_row(zero_shot_summary, lang) for lang in LANG_ORDER}),
        FEW_SHOT_KEY: macro_row({lang: extract_row(few_shot_summary, lang) for lang in LANG_ORDER}),
    }


def _collect_data() -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    registry = load_registry(REGISTRY_PATH)
    data: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {}
    for model_key in MODEL_KEYS:
        model_dir = registry[model_key]["model_dir"]
        baseline_dir = BASELINE_DIRS[model_key]
        data[model_key] = {
            BLOCK_BASE: qwen_base_rows(model_dir, baseline_dir),
            BLOCK_LORA: qwen_lora_rows(registry, model_key, model_dir),
        }
    return data


def build_few_shot_ablation_qwen_figure(project_root: Path) -> Figure:
    apply_poster_style()
    data = _collect_data()

    fig = plt.figure(figsize=FIG_SIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.16], wspace=0.06)
    gs_charts = gs[0].subgridspec(2, 2, wspace=0.2, hspace=0.55)
    axes = np.array(
        [[fig.add_subplot(gs_charts[r, c]) for c in range(2)] for r in range(2)]
    )
    legend_ax = fig.add_subplot(gs[1])
    legend_ax.axis("off")
    fig.suptitle(TITLE, fontsize=16, fontweight="bold", color=POSTER_BLUE, y=0.98)

    for r, model_key in enumerate(MODEL_KEYS):
        for c, block in enumerate(BLOCK_ORDER):
            ax = axes[r, c]
            block_data = data[model_key][block]
            plot_grouped_bars(
                ax,
                group_keys=METRIC_ORDER,
                group_labels=METRIC_LABELS,
                series_keys=SHOT_ORDER,
                series_labels=SHOT_LABELS,
                value_fn=lambda shot, metric, bd=block_data: bd[shot][metric],
                ylabel="Score",
                panel_title=f"{MODEL_LABELS[model_key]} — {BLOCK_TITLES[block]}",
                ylim_top=YLIM_TOP,
                yticks=list(range(0, YLIM_TOP + 1, 20)),
            )
            ax.tick_params(axis="x", labelsize=11)

    legend_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=SHOT_LABELS[k])
        for k in SHOT_ORDER
    ]
    legend_ax.legend(
        handles=legend_handles,
        loc="center",
        ncol=1,
        framealpha=0.95,
        title="Prompting",
        title_fontsize=14,
        fontsize=12,
        labelspacing=0.9,
        borderpad=0.8,
    )

    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.06, right=0.98)
    return fig
