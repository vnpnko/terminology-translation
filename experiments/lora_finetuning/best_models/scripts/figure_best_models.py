"""Best LoRA config (Qwen 7B, 2 epochs, zero-shot) vs. GPT (few-shot), by language pair."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "scripts"))

from shared.lib.analysis.grouped_bar_figure_common import (
    BLEU_YLIM_TOP,
    EXPANSION_COLORS,
    REPORT_BAR_WIDTH_RATIO,
    REPORT_GROUP_WIDTH,
    REPORT_LEGEND_ROW_FONTSIZE,
    REPORT_LEGEND_ROW_NCOL,
    REPORT_STACK_FIG_SIZE,
    REPORT_STACK_HSPACE,
    REPORT_SUPTITLE_FONTSIZE,
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_stack_layout,
    place_legend_row,
    plot_grouped_bars,
)
from shared.lib.analysis.metrics_loader import LANG_LABELS, LANG_ORDER
from shared.lib.analysis.plot_style import apply_report_style
from compare_common import (  # noqa: E402
    RUN_GPT_BASE,
    RUN_LORA_2_EPOCH_ZERO_SHOT,
    extract_row,
    load_metrics_summary,
    load_registry,
)

FINETUNING_RESULTS = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "results"
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "run_registry.json"
MODEL_KEY = "7B"

GPT_KEY = "gpt_few_shot"
LORA_KEY = "qwen_lora_2ep"
SERIES_ORDER = (GPT_KEY, LORA_KEY)
SERIES_LABELS = {
    GPT_KEY: "GPT (few-shot)",
    LORA_KEY: "Qwen 7B LoRA 2ep (zero-shot)",
}

# extract_row's index into compare_common.METRICS: 0=bleu, 2=term_accuracy_pct
BLEU_IDX = 0
TERM_ACC_IDX = 2

def _collect_data() -> dict[str, dict[str, list[float | None]]]:
    registry = load_registry(REGISTRY_PATH)
    model_dir = registry[MODEL_KEY]["model_dir"]
    lora_run = next(run for run in registry[MODEL_KEY]["runs"] if run["run_id"] == RUN_LORA_2_EPOCH_ZERO_SHOT)

    gpt_summary = load_metrics_summary(FINETUNING_RESULTS / RUN_GPT_BASE / "metrics_summary.json")
    lora_summary = load_metrics_summary(
        FINETUNING_RESULTS / model_dir / lora_run["folder"] / "metrics_summary.json"
    )
    return {
        GPT_KEY: {lang: extract_row(gpt_summary, lang) for lang in LANG_ORDER},
        LORA_KEY: {lang: extract_row(lora_summary, lang) for lang in LANG_ORDER},
    }


def build_best_models_figure(project_root: Path) -> Figure:
    apply_report_style()
    data = _collect_data()

    fig, axes, _ = create_pair_figure(
        None,
        figsize=REPORT_STACK_FIG_SIZE,
        hspace=REPORT_STACK_HSPACE,
        legend_position="bottom",
        stacked=True,
    )

    metric_axes = [
        (axes[0], BLEU_IDX, "BLEU", BLEU_YLIM_TOP, list(range(0, BLEU_YLIM_TOP + 1, 20))),
        (axes[1], TERM_ACC_IDX, "Term accuracy (%)", TERM_ACC_YLIM_TOP, list(range(0, TERM_ACC_YLIM_TOP + 1, 25))),
    ]
    for ax, metric_idx, ylabel, ylim_top, yticks in metric_axes:
        plot_grouped_bars(
            ax,
            group_keys=LANG_ORDER,
            group_labels=LANG_LABELS,
            series_keys=SERIES_ORDER,
            series_labels=SERIES_LABELS,
            value_fn=lambda series, lang, mi=metric_idx: data[series][lang][mi],
            ylabel=ylabel,
            ylim_top=ylim_top,
            yticks=yticks,
            panel_title_fontsize=REPORT_SUPTITLE_FONTSIZE,
            show_values=False,
            group_width=REPORT_GROUP_WIDTH,
            bar_width_ratio=REPORT_BAR_WIDTH_RATIO,
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
