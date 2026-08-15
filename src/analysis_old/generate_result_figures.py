"""Generate poster result figures from metrics_summary.json files.

Writes a PDF and PNG per figure to ``--output-dir`` (default:
``<project-root>/poster/figures``). Reads ``metrics_summary.json`` files
under ``<project-root>/results`` (and ``experiments/05_lora_finetuning``
for exp5).

Usage::

    python src/analysis/generate_result_figures.py
    python src/analysis/generate_result_figures.py --only exp1 exp5
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

from figure_exp1 import build_exp1_figure
from figure_exp23 import build_exp23_figure
from figure_exp4 import build_exp4_figure
from figure_exp5 import build_exp5_figure
from plot_style import save_figure

FIGURE_BUILDERS = {
    "exp1": ("fig_exp1_term_expansion", lambda root: build_exp1_figure(root / "results")),
    "exp23": ("fig_exp23_expansion_strategies", lambda root: build_exp23_figure(root / "results")),
    "exp4": ("fig_exp4_dev_v1_vs_dev_v2_training", lambda root: build_exp4_figure(root / "results")),
    "exp5": ("fig_exp5_lora_finetuning", lambda root: build_exp5_figure(root)),
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
