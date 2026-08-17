"""Fine-tuning Qwen models: LoRA epoch ablation vs GPT-4o-mini."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.figure_common import place_side_legend
from src.analysis.metrics_loader import LANG_LABELS, LANG_ORDER, require_paths
from src.analysis.plot_style import (
    LANG_COLORS,
    POSTER_BLUE,
    SIZE_COLORS,
    apply_poster_style,
    headroom_ylim,
)
from metrics_parser import LangMetrics, extract_proper_term_metrics, load_summary, macro_average

FINETUNING_DIR_NAME = "experiments/04_lora_finetuning"

LORA_EPOCH_RUNS = (
    (1, "qwen_lora_no_few_shots"),
    (2, "qwen_lora_no_few_shots_2_epochs"),
    (3, "qwen_lora_no_few_shots_3_epochs"),
)

NO_FEW_SHOT_PATHS = {
    "Qwen2.5-7B": "results/dev_v1/original/no-few-shots/qwen_7b/metrics_summary.json",
    "Qwen2.5-3B": "results/dev_v1/original/no-few-shots/qwen_3b/metrics_summary.json",
}

NO_FEW_SHOT_GPT_PATH = "results/dev_v1/original/no-few-shots/gpt/metrics_summary.json"

# (metrics path, label, use_finetuning_results_root)
COMPARE_RUNS = (
    (NO_FEW_SHOT_GPT_PATH, "GPT-4o-mini (no few-shot)", False),
    ("gpt_base/metrics_summary.json", "GPT-4o-mini (few-shot)", True),
    ("Qwen2.5-7B/qwen_lora_no_few_shots_2_epochs/metrics_summary.json", "Qwen 7B LoRA 2ep (no few-shot)", True),
)

# x=0: few-shot base (standalone); x=1: no-few-shot base; x=2–4: LoRA epochs 1–3.
EPOCH_XTICKS = ["0\n(few-shot)", "0", "1", "2", "3"]

TITLE = (
    "LoRA Epoch Ablation for Qwen2.5\n"
    "(train dev_v2; test dev_v1; proper_term; compared with GPT-4o-mini)"
)


def _load_run_metrics(project_root: Path, rel_path: str):
    summary = load_summary(project_root / FINETUNING_DIR_NAME / "results" / rel_path)
    by_lang = extract_proper_term_metrics(summary)
    return by_lang, macro_average(by_lang)


def _load_metrics_at_path(project_root: Path, path: Path):
    summary = load_summary(path)
    by_lang = extract_proper_term_metrics(summary)
    return by_lang, macro_average(by_lang)


def _collect_epoch_series(project_root: Path, model_dir: str) -> dict:
    finetuning_results = project_root / FINETUNING_DIR_NAME / "results" / model_dir
    paths = [
        finetuning_results / "qwen_base" / "metrics_summary.json",
        project_root / NO_FEW_SHOT_PATHS[model_dir],
        *(
            finetuning_results / folder / "metrics_summary.json"
            for _, folder in LORA_EPOCH_RUNS
        ),
    ]
    require_paths(paths)

    _, fewshot_macro = _load_run_metrics(project_root, f"{model_dir}/qwen_base/metrics_summary.json")
    _, nofs_macro = _load_metrics_at_path(project_root, project_root / NO_FEW_SHOT_PATHS[model_dir])

    bleu_line: list[tuple[int, float | None]] = [(1, nofs_macro.bleu)]
    term_line: list[tuple[int, float | None]] = [(1, nofs_macro.term_avg_pct)]

    for epoch, folder in LORA_EPOCH_RUNS:
        _, macro = _load_run_metrics(project_root, f"{model_dir}/{folder}/metrics_summary.json")
        x_pos = epoch + 1
        bleu_line.append((x_pos, macro.bleu))
        term_line.append((x_pos, macro.term_avg_pct))

    return {
        "fewshot_base": {"bleu": fewshot_macro.bleu, "term_acc": fewshot_macro.term_avg_pct},
        "line": {"bleu": bleu_line, "term_acc": term_line},
    }


def _short_model_label(label: str) -> str:
    return label.replace(" (few-shot)", "\n(few-shot)").replace(" (no few-shot)", "\n(no few-shot)")


def _compare_run_path(project_root: Path, rel_path: str, use_finetuning_results: bool) -> Path:
    if use_finetuning_results:
        return project_root / FINETUNING_DIR_NAME / "results" / rel_path
    return project_root / rel_path


def _plot_model_lang_bars(
    ax,
    compare_data: dict,
    model_labels: list[str],
    LangMetrics,
    metric_attr: str,
    ylabel: str,
) -> None:
    n_models = len(model_labels)
    n_langs = len(LANG_ORDER)
    x = np.arange(n_models)
    group_width = 0.72
    bar_width = group_width / n_langs
    all_values: list[float | None] = []

    for lang_idx, lang in enumerate(LANG_ORDER):
        offset = (lang_idx - (n_langs - 1) / 2) * bar_width
        values = [
            getattr(compare_data[label].get(lang, LangMetrics()), metric_attr)
            for label in model_labels
        ]
        all_values.extend(values)
        bars = ax.bar(
            x + offset,
            values,
            bar_width * 0.92,
            label=LANG_LABELS[lang],
            color=LANG_COLORS[lang],
            edgecolor="#333333",
            linewidth=0.6,
        )
        for bar, val in zip(bars, values):
            if val is not None:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    val,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([_short_model_label(label) for label in model_labels])
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel(ylabel, labelpad=10)
    headroom_ylim(ax, all_values)


def build_lora_finetuning_figure(project_root: Path) -> Figure:
    apply_poster_style()

    compare_paths = [
        _compare_run_path(project_root, rel_path, use_finetuning_results)
        for rel_path, _, use_finetuning_results in COMPARE_RUNS
    ]
    require_paths(compare_paths)

    series_7b = _collect_epoch_series(project_root, "Qwen2.5-7B")
    series_3b = _collect_epoch_series(project_root, "Qwen2.5-3B")

    compare_data = {}
    for rel_path, label, use_finetuning_results in COMPARE_RUNS:
        summary = load_summary(_compare_run_path(project_root, rel_path, use_finetuning_results))
        compare_data[label] = extract_proper_term_metrics(summary)

    fig = plt.figure(figsize=(17, 11))
    gs_outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 0.14],
        height_ratios=[1, 1.05],
        hspace=0.58,
        wspace=0.08,
    )
    gs_top = gs_outer[0, 0].subgridspec(1, 2, wspace=0.26)
    gs_bottom = gs_outer[1, 0].subgridspec(1, 2, wspace=0.22)
    ax_bleu = fig.add_subplot(gs_top[0, 0])
    ax_term = fig.add_subplot(gs_top[0, 1])
    ax_compare_bleu = fig.add_subplot(gs_bottom[0, 0])
    ax_compare_term = fig.add_subplot(gs_bottom[0, 1])
    ax_legend_top = fig.add_subplot(gs_outer[0, 1])
    ax_legend_bottom = fig.add_subplot(gs_outer[1, 1])
    ax_legend_top.axis("off")
    ax_legend_bottom.axis("off")

    fig.suptitle(TITLE, fontsize=16, fontweight="bold", color=POSTER_BLUE, y=0.97)

    line_styles = {
        "7B": {
            "color": SIZE_COLORS["7B"],
            "marker": "o",
            "label": "Qwen 7B",
            "annotate_offset": (0, 10),
            "annotate_va": "bottom",
        },
        "3B": {
            "color": SIZE_COLORS["3B"],
            "marker": "s",
            "label": "Qwen 3B",
            "annotate_offset": (0, -10),
            "annotate_va": "top",
        },
    }
    series_by_size = {"7B": series_7b, "3B": series_3b}

    for ax, metric_key, ylabel in [
        (ax_bleu, "bleu", "BLEU (macro avg)"),
        (ax_term, "term_acc", "Term accuracy % (macro avg)"),
    ]:
        panel_values: list[float | None] = []
        for size, series in series_by_size.items():
            style = line_styles[size]
            points = series["line"][metric_key]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            panel_values.extend(ys)

            fewshot_val = series["fewshot_base"][metric_key]
            if fewshot_val is not None:
                panel_values.append(fewshot_val)
                ax.plot(
                    [0],
                    [fewshot_val],
                    linestyle="none",
                    marker=style["marker"],
                    color=style["color"],
                    markersize=9,
                )
                ax.annotate(
                    f"{fewshot_val:.1f}",
                    (0, fewshot_val),
                    textcoords="offset points",
                    xytext=style["annotate_offset"],
                    ha="center",
                    va=style["annotate_va"],
                    fontsize=10,
                    fontweight="bold",
                )

            ax.plot(
                xs,
                ys,
                marker=style["marker"],
                color=style["color"],
                linewidth=2.8,
                markersize=9,
                label=style["label"],
            )
            for x, y in points:
                if y is not None:
                    ax.annotate(
                        f"{y:.1f}",
                        (x, y),
                        textcoords="offset points",
                        xytext=style["annotate_offset"],
                        ha="center",
                        va=style["annotate_va"],
                        fontsize=10,
                        fontweight="bold",
                    )

        ax.axvline(3, color="#888888", linestyle="--", linewidth=1.4, alpha=0.85, zorder=0)
        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(EPOCH_XTICKS)
        ax.set_xlabel("Epochs", labelpad=12)
        ax.set_ylabel(ylabel, labelpad=10)
        headroom_ylim(ax, panel_values)

    model_labels = [label for _, label, _ in COMPARE_RUNS]
    _plot_model_lang_bars(
        ax_compare_bleu,
        compare_data,
        model_labels,
        LangMetrics,
        "bleu",
        "BLEU",
    )
    _plot_model_lang_bars(
        ax_compare_term,
        compare_data,
        model_labels,
        LangMetrics,
        "term_avg_pct",
        "Term accuracy (%)",
    )

    size_handles = [
        Line2D(
            [0],
            [0],
            marker=line_styles[size]["marker"],
            color=line_styles[size]["color"],
            linewidth=2.8,
            markersize=9,
            label=line_styles[size]["label"],
        )
        for size in ("7B", "3B")
    ]
    place_side_legend(ax_legend_top, size_handles, "Model size")

    lang_handles = [
        Patch(facecolor=LANG_COLORS[lang], edgecolor="#333333", label=LANG_LABELS[lang])
        for lang in LANG_ORDER
    ]
    place_side_legend(ax_legend_bottom, lang_handles, "Language pair")

    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.07, right=0.98)
    return fig
