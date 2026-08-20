"""LoRA epoch ablation for Qwen2.5: BLEU/term accuracy vs. epoch count, both model sizes."""

from __future__ import annotations

import sys
from pathlib import Path

from matplotlib.figure import Figure
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "scripts"))

from shared.lib.analysis.figure_common import BLEU_YLIM_TOP, TERM_ACC_YLIM_TOP, create_pair_figure, finalize_pair_layout, place_side_legend
from shared.lib.analysis.metrics_loader import require_paths
from shared.lib.analysis.plot_style import SIZE_COLORS, apply_poster_style, fixed_top_ylim
from metrics_parser import load_summary, extract_proper_term_metrics, macro_average  # noqa: E402

FINETUNING_RESULTS = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "results"

LORA_EPOCH_RUNS = (
    (1, "qwen_lora_no_few_shots"),
    (2, "qwen_lora_no_few_shots_2_epochs"),
    (3, "qwen_lora_no_few_shots_3_epochs"),
)

ZERO_SHOT_PATHS = {
    "Qwen2.5-7B": "shared/results/dev_v1/original/zero_shot/qwen_7b/metrics_summary.json",
    "Qwen2.5-3B": "shared/results/dev_v1/original/zero_shot/qwen_3b/metrics_summary.json",
}

MODEL_STYLES = {
    "Qwen2.5-7B": {"color": SIZE_COLORS["7B"], "marker": "o", "label": "Qwen 7B"},
    "Qwen2.5-3B": {"color": SIZE_COLORS["3B"], "marker": "s", "label": "Qwen 3B"},
}

TITLE = (
    "LoRA epoch ablation for Qwen2.5\n"
    "(train dev_v2; test dev_v1; proper_term; zero-shot)"
)


def _collect_epoch_series(project_root: Path, model_dir: str) -> list[tuple[int, float | None, float | None]]:
    zero_shot_summary = load_summary(project_root / ZERO_SHOT_PATHS[model_dir])
    zero_shot_macro = macro_average(extract_proper_term_metrics(zero_shot_summary))
    points = [(0, zero_shot_macro.bleu, zero_shot_macro.term_avg_pct)]

    for epoch, folder in LORA_EPOCH_RUNS:
        summary = load_summary(FINETUNING_RESULTS / model_dir / folder / "metrics_summary.json")
        macro = macro_average(extract_proper_term_metrics(summary))
        points.append((epoch, macro.bleu, macro.term_avg_pct))

    return points


def build_epoch_ablation_figure(project_root: Path) -> Figure:
    apply_poster_style()

    paths = [project_root / ZERO_SHOT_PATHS[model_dir] for model_dir in MODEL_STYLES]
    paths.extend(
        FINETUNING_RESULTS / model_dir / folder / "metrics_summary.json"
        for model_dir in MODEL_STYLES
        for _, folder in LORA_EPOCH_RUNS
    )
    require_paths(paths)

    series = {model_dir: _collect_epoch_series(project_root, model_dir) for model_dir in MODEL_STYLES}

    fig, axes, legend_ax = create_pair_figure(TITLE)

    metric_axes = [
        (axes[0], 1, "BLEU (macro avg)", BLEU_YLIM_TOP, list(range(0, BLEU_YLIM_TOP + 1, 10))),
        (axes[1], 2, "Term accuracy (%) (macro avg)", TERM_ACC_YLIM_TOP, list(range(0, TERM_ACC_YLIM_TOP + 1, 20))),
    ]
    for ax, value_idx, ylabel, ylim_top, yticks in metric_axes:
        for model_dir, style in MODEL_STYLES.items():
            points = series[model_dir]
            xs = [p[0] for p in points]
            ys = [p[value_idx] for p in points]
            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                color=style["color"],
                linewidth=2.8,
                markersize=9,
                label=style["label"],
            )
            for x, y in zip(xs, ys):
                if y is not None:
                    ax.annotate(
                        f"{y:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        fontweight="bold",
                    )
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xlabel("Epochs", labelpad=12)
        ax.set_ylabel(ylabel, labelpad=10)
        fixed_top_ylim(ax, ylim_top)
        ax.set_yticks(yticks)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=style["marker"],
            color=style["color"],
            linewidth=2.8,
            markersize=9,
            label=style["label"],
        )
        for style in MODEL_STYLES.values()
    ]
    place_side_legend(legend_ax, legend_handles, "Model size")
    finalize_pair_layout(fig)
    return fig
