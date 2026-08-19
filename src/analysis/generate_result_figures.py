"""Generate result figures from metrics_summary.json files.

Writes a PDF and PNG per figure into its owning experiment's ``figures/``
directory by default (e.g. ``experiments/term_expansion/by_model/figures/``)
— that's each figure's canonical home. ``--output-dir`` overrides this and
writes every selected figure into one shared directory instead, useful for
ad hoc regeneration. Note that ``poster/figures/`` and ``report/figures/``
are curated, manually-copied collections of whichever figures are actually
used in the poster/paper (not generation targets); after regenerating a
figure's home copy, copy it over by hand if it needs to be updated there.
Reads ``metrics_summary.json`` files under ``<project-root>/results`` (and
``experiments/lora_finetuning`` for lora_finetuning).

Usage::

    python src/analysis/generate_result_figures.py
    python src/analysis/generate_result_figures.py --only model_comparison lora_finetuning
    python src/analysis/generate_result_figures.py --output-dir /tmp/figures
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
    PROJECT_ROOT / "experiments" / "term_expansion" / "by_model" / "scripts",
    PROJECT_ROOT / "experiments" / "term_expansion" / "by_language_pair" / "scripts",
    PROJECT_ROOT / "experiments" / "term_expansion" / "dataset_comparison" / "scripts",
    PROJECT_ROOT / "experiments" / "lora_finetuning" / "scripts",
]
for _scripts_dir in EXPERIMENT_SCRIPTS_DIRS:
    sys.path.insert(0, str(_scripts_dir))

from figure_model_comparison import build_model_comparison_figure
from figure_mode_comparison import build_mode_comparison_figure
from figure_dataset_comparison import build_dataset_comparison_figure
from figure_lora_finetuning import build_lora_finetuning_figure
from plot_style import save_figure

FIGURE_BUILDERS = {
    "model_comparison": (
        "fig_term_expansion",
        lambda root: build_model_comparison_figure(root / "results"),
        Path("experiments/term_expansion/by_model/figures"),
    ),
    "mode_comparison": (
        "fig_expansion_strategies",
        lambda root: build_mode_comparison_figure(root / "results"),
        Path("experiments/term_expansion/by_language_pair/figures"),
    ),
    "dataset_comparison": (
        "fig_dev_v1_vs_dev_v2_training",
        lambda root: build_dataset_comparison_figure(root / "results"),
        Path("experiments/term_expansion/dataset_comparison/figures"),
    ),
    "lora_finetuning": (
        "fig_lora_finetuning",
        lambda root: build_lora_finetuning_figure(root),
        Path("experiments/lora_finetuning/figures"),
    ),
}


def generate_figures(
    project_root: Path,
    output_dir: Path | None = None,
    only: list[str] | None = None,
) -> list[tuple[str, Path, Path]]:
    keys = only if only else list(FIGURE_BUILDERS.keys())
    unknown = [k for k in keys if k not in FIGURE_BUILDERS]
    if unknown:
        raise SystemExit(f"Unknown figure keys: {unknown}. Choose from: {list(FIGURE_BUILDERS)}")

    written: list[tuple[str, Path, Path]] = []
    for key in keys:
        stem, builder, home_dir = FIGURE_BUILDERS[key]
        fig = builder(project_root)
        target_dir = output_dir if output_dir is not None else project_root / home_dir
        pdf_path, png_path = save_figure(fig, target_dir, stem)
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
        help=(
            "Write every selected figure into this one directory instead of its "
            "owning experiment's figures/ home (default: each figure's home dir)"
        ),
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
    output_dir = args.output_dir.resolve() if args.output_dir else None

    written = generate_figures(project_root, output_dir, args.only)
    if output_dir is not None:
        print(f"Done. Figures saved to {output_dir}")
    else:
        for stem, pdf_path, _ in written:
            print(f"{stem}: saved to {pdf_path.parent}")


if __name__ == "__main__":
    main()
