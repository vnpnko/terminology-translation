"""Export finetuning experiment metrics to Excel workbooks for reporting.

Usage:
    python experiments/04_lora_finetuning/scripts/export_finetuning_report.py
    python experiments/04_lora_finetuning/scripts/export_finetuning_report.py --model 7B
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from metrics_parser import load_all_runs, load_run
from sheet_builders import (
    build_base_vs_lora,
    build_epoch_ablation,
    build_experiment_config,
    build_gpt_baseline,
    build_gpt_vs_best_qwen,
    build_main_results,
    build_training_loss,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "results"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "report"
REGISTRY_PATH = SCRIPT_DIR / "run_registry.json"


def load_registry() -> dict:
    with REGISTRY_PATH.open(encoding="utf-8") as f:
        return json.load(f)


def write_qwen_workbook(
    model_cfg: dict,
    runs: list,
    gpt_run,
    output_dir: Path,
) -> Path:
    sheets = {
        "main_results": build_main_results(runs, gpt_run),
        "base_vs_lora": build_base_vs_lora(runs),
        "epoch_ablation": build_epoch_ablation(runs),
        "experiment_config": build_experiment_config(runs),
    }
    loss_wide, loss_meta = build_training_loss(runs)

    out_path = output_dir / model_cfg["workbook"]
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)
        loss_meta.to_excel(writer, sheet_name="training_loss_meta", index=False)
        loss_wide.to_excel(writer, sheet_name="training_loss", index=False)

    return out_path


def write_gpt_workbook(
    gpt_cfg: dict,
    gpt_run,
    qwen_runs_by_size: dict[str, list],
    output_dir: Path,
) -> Path:
    sheets = {
        "gpt_baseline": build_gpt_baseline(gpt_run),
        "gpt_vs_best_qwen": build_gpt_vs_best_qwen(gpt_run, qwen_runs_by_size),
        "experiment_config": build_experiment_config([gpt_run]),
    }

    out_path = output_dir / gpt_cfg["workbook"]
    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name, index=False)

    return out_path


def export_all(results_dir: Path, output_dir: Path, model_filter: str | None = None) -> list[Path]:
    registry = load_registry()
    output_dir.mkdir(parents=True, exist_ok=True)

    gpt_meta = registry["gpt"]["runs"][0]
    gpt_run = load_run(gpt_meta, results_dir)

    written: list[Path] = []
    qwen_runs_by_size: dict[str, list] = {}

    for model_key in ("3B", "7B"):
        if model_filter and model_filter != model_key:
            continue
        model_cfg = registry[model_key]
        run_dir = results_dir / model_cfg["model_dir"]
        runs = load_all_runs(model_cfg["runs"], run_dir)
        qwen_runs_by_size[model_key] = runs
        written.append(write_qwen_workbook(model_cfg, runs, gpt_run, output_dir))

    if model_filter is None or model_filter == "gpt":
        if model_filter == "gpt" and not qwen_runs_by_size:
            for model_key in ("3B", "7B"):
                model_cfg = registry[model_key]
                run_dir = results_dir / model_cfg["model_dir"]
                qwen_runs_by_size[model_key] = load_all_runs(model_cfg["runs"], run_dir)
        written.append(
            write_gpt_workbook(registry["gpt"], gpt_run, qwen_runs_by_size, output_dir)
        )

    return written


def main() -> int:
    parser = argparse.ArgumentParser(description="Export finetuning experiment results to Excel.")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=DEFAULT_RESULTS_DIR,
        help="Path to experiments/04_lora_finetuning/results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for output .xlsx files",
    )
    parser.add_argument(
        "--model",
        choices=("3B", "7B", "gpt"),
        default=None,
        help="Export only one model workbook (default: all three)",
    )
    args = parser.parse_args()

    try:
        written = export_all(args.results_dir, args.output_dir, args.model)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    print("Wrote:")
    for path in written:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
