"""Poster-oriented matplotlib styling for result figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

POSTER_BLUE = "#2E527D"
POSTER_ACCENT = "#A9BCD1"
POSTER_BG = "#E8EEF3"

# Distinct hues for easy comparison (not monochrome blue).
COLOR_GPT = "#264653"
COLOR_QWEN_3B = "#E76F51"
COLOR_QWEN_7B = "#2A9D8F"
COLOR_ORIGINAL = "#6C757D"
COLOR_NO_TERM = "#CED4DA"
COLOR_RANDOM_TERM = "#F8CBAD"
COLOR_GPT_EXPAND = "#F4A261"
COLOR_DICTIONARY = "#7209B7"
COLOR_EXTERNAL_DICTIONARY = "#9D4EDD"
COLOR_BLEU = "#457B9D"
COLOR_CHRF = "#E9C46A"
COLOR_TERM_ACC = "#2A9D8F"
COLOR_WEIGHTED_CONS = "#9B2226"
COLOR_QWEN_BASE = "#E9C46A"
COLOR_LORA = "#E76F51"

MODEL_COLORS = {
    "GPT": COLOR_GPT,
    "Qwen 3B": COLOR_QWEN_3B,
    "Qwen 7B": COLOR_QWEN_7B,
    "GPT-4o-mini": COLOR_GPT,
    "GPT-4o-mini (few-shot)": COLOR_GPT,
    "Qwen 7B base": COLOR_QWEN_BASE,
    "Qwen 7B base (few-shot)": COLOR_QWEN_BASE,
    "Qwen 7B LoRA 2ep": COLOR_LORA,
    "Qwen 7B LoRA 2ep (no few-shot)": COLOR_LORA,
}

STRATEGY_COLORS = {
    "Original": COLOR_ORIGINAL,
    "GPT expand": COLOR_GPT_EXPAND,
    "Dictionary": COLOR_DICTIONARY,
}

VARIANT_COLORS = {
    "Original": COLOR_ORIGINAL,
    "GPT-expanded": COLOR_GPT_EXPAND,
    "Domain-filtered": COLOR_DICTIONARY,
}

METRIC_COLORS = {
    "BLEU": COLOR_BLEU,
    "chrF": COLOR_CHRF,
    "Term Accuracy %": COLOR_TERM_ACC,
    "Weighted Consistency": COLOR_WEIGHTED_CONS,
}

SIZE_COLORS = {
    "7B": COLOR_QWEN_7B,
    "3B": COLOR_QWEN_3B,
}

LANG_COLORS = {
    "ende": COLOR_BLEU,
    "enru": COLOR_GPT_EXPAND,
    "enes": COLOR_QWEN_7B,
}

FIG_DPI = 300


def apply_poster_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 15,
            "axes.titlesize": 15,
            "axes.labelsize": 14,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#333333",
            "axes.labelcolor": "#222222",
            "axes.grid": True,
            "grid.alpha": 0.3,
            "grid.color": "#cccccc",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def headroom_ylim(ax, values: list[float | None], pad_ratio: float = 0.14) -> None:
    present = [v for v in values if v is not None]
    if not present:
        ax.set_ylim(bottom=0)
        return
    ymax = max(present)
    ax.set_ylim(0, ymax * (1 + pad_ratio))


def fixed_top_ylim(ax, top: float, pad_ratio: float = 0.06) -> None:
    """Fix the y-axis top to ``top``, with a small pad for bar value labels."""
    ax.set_ylim(0, top * (1 + pad_ratio))


def save_figure(fig: Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    save_kw = {"dpi": FIG_DPI, "bbox_inches": "tight", "pad_inches": 0.08}

    fig.savefig(pdf_path, facecolor="white", **save_kw)

    # PNG: transparent figure background; axes stay white for readability on posters.
    fig.patch.set_facecolor("none")
    fig.savefig(png_path, transparent=True, **save_kw)
    fig.patch.set_facecolor("white")

    return pdf_path, png_path
