"""Fine-tuning Qwen models: LoRA epoch ablation vs GPT-4o-mini."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from figure_common import (
    CAPTION_FONTSIZE,
    CAPTION_X,
    CAPTION_Y,
    place_side_legend,
)
from metrics_loader import LANG_LABELS, LANG_ORDER, require_paths
from plot_style import (
    LANG_COLORS,
    POSTER_BLUE,
    SIZE_COLORS,
    apply_poster_style,
    headroom_ylim,
)

FINETUNING_DIR_NAME = "Finetuning experiment"

EPOCH_RUNS = (
    (0, "qwen_base"),
    (1, "qwen_lora_no_few_shots"),
    (2, "qwen_lora_no_few_shots_2_epochs"),
    (3, "qwen_lora_no_few_shots_3_epochs"),
)

COMPARE_RUNS = (
    ("gpt_base/metrics_summary.json", "GPT-4o-mini (few-shot)"),
    ("Qwen2.5-7B/qwen_base/metrics_summary.json", "Qwen 7B base (few-shot)"),
    ("Qwen2.5-7B/qwen_lora_no_few_shots_2_epochs/metrics_summary.json", "Qwen 7B LoRA 2ep (no few-shot)"),
)

# Epoch 0 = untuned base with 3-shot prompt; epochs 1–3 = LoRA without few-shot (ExperimentsSummary §5).
EPOCH_XTICKS = ["0\n(base,\nfew-shot)", "1", "2", "3"]

TITLE = (
    "Fine-Tuning Qwen2.5 on SAP dev_v2: LoRA Epoch Ablation vs GPT-4o-mini\n"
    "(train ~1,500 sent./lang · test held-out dev_v1 · proper_term)"
)
SUBTITLE = (
    "Epoch 0: untuned Qwen base with 3-shot prompt; epochs 1–3: LoRA without few-shot. "
    "LoRA closes the BLEU gap on EN→ES/RU; GPT-4o-mini remains strongest for terminology accuracy."
)


def _finetuning_scripts_dir(project_root: Path) -> Path:
    return project_root / FINETUNING_DIR_NAME / "scripts"


def _import_finetuning_parser(project_root: Path):
    scripts_dir = _finetuning_scripts_dir(project_root)
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from metrics_parser import LangMetrics, extract_proper_term_metrics, load_summary, macro_average

    return LangMetrics, extract_proper_term_metrics, load_summary, macro_average


def _load_run_metrics(project_root: Path, rel_path: str):
    _, extract_proper_term_metrics, load_summary, macro_average = _import_finetuning_parser(
        project_root
    )
    summary = load_summary(project_root / FINETUNING_DIR_NAME / "results" / rel_path)
    by_lang = extract_proper_term_metrics(summary)
    return by_lang, macro_average(by_lang)


def _collect_epoch_series(project_root: Path, model_dir: str) -> dict[str, list[tuple[int, float | None]]]:
    paths = [
        project_root
        / FINETUNING_DIR_NAME
        / "results"
        / model_dir
        / folder
        / "metrics_summary.json"
        for _, folder in EPOCH_RUNS
    ]
    require_paths(paths)

    bleu_series: list[tuple[int, float | None]] = []
    term_series: list[tuple[int, float | None]] = []

    for epochs, folder in EPOCH_RUNS:
        _, macro = _load_run_metrics(project_root, f"{model_dir}/{folder}/metrics_summary.json")
        bleu_series.append((epochs, macro.bleu))
        term_series.append((epochs, macro.term_avg_pct))

    return {"bleu": bleu_series, "term_acc": term_series}


def _short_model_label(label: str) -> str:
    return label.replace(" (few-shot)", "\n(few-shot)").replace(" (no few-shot)", "\n(no few-shot)")


def _plot_model_lang_bars(
    ax,
    compare_data: dict,
    model_labels: list[str],
    LangMetrics,
    metric_attr: str,
    ylabel: str,
    title: str,
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
    ax.set_title(title, loc="left", fontsize=14, color=POSTER_BLUE, pad=16)
    headroom_ylim(ax, all_values)


def build_exp5_figure(project_root: Path) -> Figure:
    apply_poster_style()

    compare_paths = [
        project_root / FINETUNING_DIR_NAME / "results" / rel for rel, _ in COMPARE_RUNS
    ]
    require_paths(compare_paths)

    series_7b = _collect_epoch_series(project_root, "Qwen2.5-7B")
    series_3b = _collect_epoch_series(project_root, "Qwen2.5-3B")

    LangMetrics, extract_proper_term_metrics, load_summary, _ = _import_finetuning_parser(
        project_root
    )
    compare_data = {}
    for rel, label in COMPARE_RUNS:
        summary = load_summary(project_root / FINETUNING_DIR_NAME / "results" / rel)
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
        "7B": {"color": SIZE_COLORS["7B"], "marker": "o", "label": "Qwen2.5-7B-Instruct"},
        "3B": {"color": SIZE_COLORS["3B"], "marker": "s", "label": "Qwen2.5-3B-Instruct"},
    }
    series_by_size = {"7B": series_7b, "3B": series_3b}

    for ax, metric_key, ylabel, panel_title in [
        (ax_bleu, "bleu", "BLEU (macro avg)", "LoRA epochs vs BLEU"),
        (ax_term, "term_acc", "Term accuracy % (macro avg)", "LoRA epochs vs term accuracy"),
    ]:
        panel_values: list[float | None] = []
        for size, series in series_by_size.items():
            style = line_styles[size]
            points = series[metric_key]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            panel_values.extend(ys)
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
                        xytext=(0, 10),
                        ha="center",
                        fontsize=10,
                        fontweight="bold",
                    )

        ax.axvline(2, color="#888888", linestyle="--", linewidth=1.4, alpha=0.85)
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(EPOCH_XTICKS)
        ax.set_ylabel(ylabel, labelpad=10)
        ax.set_title(panel_title, loc="left", fontsize=14, color=POSTER_BLUE, pad=10)
        headroom_ylim(ax, panel_values)

    ax_bleu.set_xlabel("")
    ax_term.set_xlabel("LoRA fine-tuning epochs (1–3: no few-shot at inference)", labelpad=12)

    ax_bleu.text(
        2.08,
        ax_bleu.get_ylim()[1] * 0.92,
        "best overall\n(7B)",
        fontsize=10,
        color="#555555",
        va="top",
    )

    model_labels = [label for _, label in COMPARE_RUNS]
    _plot_model_lang_bars(
        ax_compare_bleu,
        compare_data,
        model_labels,
        LangMetrics,
        "bleu",
        "BLEU",
        "BLEU by model and language pair",
    )
    _plot_model_lang_bars(
        ax_compare_term,
        compare_data,
        model_labels,
        LangMetrics,
        "term_avg_pct",
        "Term accuracy (%)",
        "Term accuracy by model and language pair",
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

    fig.text(
        CAPTION_X,
        CAPTION_Y,
        SUBTITLE,
        ha="center",
        va="bottom",
        fontsize=CAPTION_FONTSIZE,
        color="#444444",
    )
    fig.subplots_adjust(top=0.82, bottom=0.14, left=0.07, right=0.98)
    return fig
