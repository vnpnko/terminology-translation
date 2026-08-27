"""Data-leakage honesty check: BLEU and Term Accuracy, grouped by model, overlap vs no-overlap."""

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
    TERM_ACC_YLIM_TOP,
    create_pair_figure,
    finalize_stack_layout,
    place_legend_row,
    plot_grouped_bars,
)
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
LANG_ORDER = ("ende", "enes", "enru")

MODEL_ORDER = ("qwen_base", "gpt", "qwen_lora")
MODEL_LABELS = {
    "qwen_base": "Qwen 7B\n(untrained)",
    "gpt": "GPT",
    "qwen_lora": "Qwen 7B\n(LoRA)",
}

OVERLAP_KEY = "overlap_data"
NO_OVERLAP_KEY = "no_overlap_data"
SUBSET_ORDER = (OVERLAP_KEY, NO_OVERLAP_KEY)
SUBSET_LABELS = {
    OVERLAP_KEY: "Overlap data",
    NO_OVERLAP_KEY: "No-overlap data",
}

# extract_row's index into compare_common.METRICS: 0=bleu, 2=term_accuracy_pct
BLEU_IDX = 0
TERM_ACC_IDX = 2

def macro_row(rows: dict[str, list[float | None]]) -> list[float | None]:
    n_metrics = len(rows[LANG_ORDER[0]])
    out: list[float | None] = []
    for i in range(n_metrics):
        vals = [rows[lang][i] for lang in LANG_ORDER if rows[lang][i] is not None]
        out.append(round(sum(vals) / len(vals), 2) if vals else None)
    return out


def subset_rows(base_dir: Path, subset: str) -> dict[str, list[float | None]]:
    summary = load_metrics_summary(base_dir / "test_cleaned_by_sentences" / subset / "metrics_summary.json")
    return {lang: extract_row(summary, lang) for lang in LANG_ORDER}


def _collect_data() -> dict[str, dict[str, list[float | None]]]:
    registry = load_registry(REGISTRY_PATH)
    model_dir = registry[MODEL_KEY]["model_dir"]
    lora_run = next(run for run in registry[MODEL_KEY]["runs"] if run["run_id"] == RUN_LORA_2_EPOCH_ZERO_SHOT)

    gpt_dir = FINETUNING_RESULTS / RUN_GPT_BASE
    base_dir = FINETUNING_RESULTS / model_dir / "qwen_base"
    lora_dir = FINETUNING_RESULTS / model_dir / lora_run["folder"]

    data: dict[str, dict[str, list[float | None]]] = {}
    for model_key, dir_path in (("qwen_base", base_dir), ("gpt", gpt_dir), ("qwen_lora", lora_dir)):
        data[model_key] = {
            OVERLAP_KEY: macro_row(subset_rows(dir_path, "overlap")),
            NO_OVERLAP_KEY: macro_row(subset_rows(dir_path, "no_overlap")),
        }
    return data


def build_leakage_check_figure(project_root: Path) -> Figure:
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
            group_keys=MODEL_ORDER,
            group_labels=MODEL_LABELS,
            series_keys=SUBSET_ORDER,
            series_labels=SUBSET_LABELS,
            value_fn=lambda subset, model, mi=metric_idx: data[model][subset][mi],
            ylabel=ylabel,
            ylim_top=ylim_top,
            yticks=yticks,
            show_values=False,
            group_width=REPORT_GROUP_WIDTH,
            bar_width_ratio=REPORT_BAR_WIDTH_RATIO,
        )

    axes[0].tick_params(labelbottom=False)

    legend_handles = [
        Patch(facecolor=EXPANSION_COLORS[k], edgecolor="#333333", label=SUBSET_LABELS[k])
        for k in SUBSET_ORDER
    ]
    place_legend_row(
        fig,
        legend_handles,
        fontsize=REPORT_LEGEND_ROW_FONTSIZE,
        ncol=REPORT_LEGEND_ROW_NCOL,
    )
    finalize_stack_layout(fig)
    return fig
