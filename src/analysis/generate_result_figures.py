"""Generate poster result figures from metrics_summary.json files.

Writes a PDF and PNG per figure to ``--output-dir`` (default:
``<project-root>/poster/figures``). Reads ``metrics_summary.json`` files
under ``<project-root>/results`` (and ``experiments/04_lora_finetuning``
for lora_finetuning).

Usage::

    python src/analysis/generate_result_figures.py
    python src/analysis/generate_result_figures.py --only model_comparison lora_finetuning
    python src/analysis/generate_result_figures.py --output-dir poster/figures
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))

EXPERIMENT_SCRIPTS_DIRS = [
    PROJECT_ROOT / "experiments" / "01_term_expansion_by_model" / "scripts",
    PROJECT_ROOT / "experiments" / "02_term_expansion_by_language_pair" / "scripts",
    PROJECT_ROOT / "experiments" / "03_dataset_comparison" / "scripts",
    PROJECT_ROOT / "experiments" / "04_lora_finetuning" / "scripts",
]
for _scripts_dir in EXPERIMENT_SCRIPTS_DIRS:
    sys.path.insert(0, str(_scripts_dir))

from figure_model_comparison import build_model_comparison_figure
from figure_mode_comparison import build_mode_comparison_figure
from figure_dataset_comparison import build_dataset_comparison_figure
from figure_lora_finetuning import build_lora_finetuning_figure
from plot_style import save_figure

FIGURE_BUILDERS = {
    "model_comparison": ("fig_term_expansion", lambda root: build_model_comparison_figure(root / "results")),
    "mode_comparison": ("fig_expansion_strategies", lambda root: build_mode_comparison_figure(root / "results")),
    "dataset_comparison": ("fig_dev_v1_vs_dev_v2_training", lambda root: build_dataset_comparison_figure(root / "results")),
    "lora_finetuning": ("fig_lora_finetuning", lambda root: build_lora_finetuning_figure(root)),
}


def generate_figures(
    project_root: Path,
    output_dir: Path,
    only: list[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    keys = only if only else list(FIGURE_BUILDERS.keys())
    unknown = [k for k in keys if k not in FIGURE_BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown figure keys: {unknown}. Choose from: {list(FIGURE_BUILDERS)}")

    written: list[tuple[str, Path, Path]] = []
    for key in keys:
        stem, builder = FIGURE_BUILDERS[key]
        fig = builder(project_root)
        pdf_path, png_path = save_figure(fig, output_dir, stem)
        plt.close(fig)
        written.append((stem, pdf_path, png_path))
        print(f"Wrote {pdf_path.name} and {png_path.name}")

    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
        help="Path to terminology-translation project root",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures (default: <project-root>/poster/figures)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        choices=list(FIGURE_BUILDERS.keys()),
        help="Generate only selected figures",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir
        else project_root / "poster" / "figures"
    )

    generate_figures(project_root, output_dir, args.only)
    print(f"Done. Figures saved to {output_dir}")


if __name__ == "__main__":
    main()
