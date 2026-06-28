"""Parse finetuning experiment metrics_summary.json and training_loss.txt files."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

LANG_ORDER = ("ende", "enes", "enru")
LANG_LABELS = {
    "ende": "en→de",
    "enes": "en→es",
    "enru": "en→ru",
}
MODE = "proper_term"


@dataclass
class LangMetrics:
    bleu: float | None = None
    chrf: float | None = None
    total_terms: int | None = None
    term_avg_pct: float | None = None
    sample_count: int | None = None


@dataclass
class RunConfig:
    run_id: str
    label: str
    folder: str
    is_baseline: bool = False
    model: str | None = None
    use_few_shot: bool | None = None
    num_epochs: int | None = None
    mode: str | None = None
    metrics_path: Path | None = None
    training_loss_path: Path | None = None
    metrics_file_exists: bool = False
    training_loss_exists: bool = False
    training_loss_populated: bool = False
    notes: str = ""


@dataclass
class ParsedRun:
    config: RunConfig
    by_lang: dict[str, LangMetrics] = field(default_factory=dict)


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_metric(metrics: dict, spec: str | tuple[str, str]) -> float | None:
    if isinstance(spec, str):
        value = metrics.get(spec)
    else:
        section, key = spec
        value = metrics.get(section, {}).get(key)
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def extract_proper_term_metrics(summary: dict) -> dict[str, LangMetrics]:
    by_lang: dict[str, LangMetrics] = {}

    for lang in LANG_ORDER:
        lang_data = summary.get("languages", {}).get(lang)
        if not lang_data:
            continue

        mode_data = lang_data.get("modes", {}).get(MODE)
        if not mode_data:
            continue

        metrics = mode_data.get("metrics", {})
        term_acc = metrics.get("terminology_accuracy", {})
        by_lang[lang] = LangMetrics(
            bleu=extract_metric(metrics, "bleu"),
            chrf=extract_metric(metrics, "chrf"),
            total_terms=term_acc.get("total_terms"),
            term_avg_pct=extract_metric(term_acc, "avg_ratio_pct"),
            sample_count=lang_data.get("sample_count"),
        )

    return by_lang


def extract_run_config_from_summary(
    summary: dict,
    run_meta: dict,
    metrics_path: Path,
    training_loss_path: Path | None,
) -> RunConfig:
    loss_exists = training_loss_path is not None and training_loss_path.exists()
    loss_populated = loss_exists and training_loss_path.stat().st_size > 0

    return RunConfig(
        run_id=run_meta["run_id"],
        label=run_meta["label"],
        folder=run_meta["folder"],
        is_baseline=run_meta.get("is_baseline", False),
        model=summary.get("model"),
        use_few_shot=summary.get("use_few_shot", run_meta.get("use_few_shot")),
        num_epochs=summary.get("num_epochs", run_meta.get("num_epochs")),
        mode=summary.get("mode", MODE),
        metrics_path=metrics_path,
        training_loss_path=training_loss_path,
        metrics_file_exists=True,
        training_loss_exists=loss_exists,
        training_loss_populated=loss_populated,
        notes=run_meta.get("notes", ""),
    )


def parse_training_loss(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame(columns=["step", "loss"])

    rows: list[tuple[int, float]] = []
    in_table = False

    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.lower().startswith("step") and "loss" in line.lower():
                in_table = True
                continue
            if not in_table:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            try:
                step = int(parts[0])
                loss = float(parts[1])
            except ValueError:
                continue
            rows.append((step, loss))

    if not rows:
        return pd.DataFrame(columns=["step", "loss"])

    return pd.DataFrame(rows, columns=["step", "loss"])


def macro_average(by_lang: dict[str, LangMetrics]) -> LangMetrics:
    langs = [by_lang[lang] for lang in LANG_ORDER if lang in by_lang]
    if not langs:
        return LangMetrics()

    def mean_attr(attr: str) -> float | None:
        values = [getattr(row, attr) for row in langs if getattr(row, attr) is not None]
        if not values:
            return None
        return sum(values) / len(values)

    total_terms = sum(row.total_terms for row in langs if row.total_terms is not None)
    return LangMetrics(
        bleu=mean_attr("bleu"),
        chrf=mean_attr("chrf"),
        total_terms=total_terms if langs else None,
        term_avg_pct=mean_attr("term_avg_pct"),
        sample_count=int(mean_attr("sample_count")) if mean_attr("sample_count") is not None else None,
    )


def validate_metrics_paths(run_metas: list[dict], run_dir: Path) -> list[str]:
    missing: list[str] = []
    for run_meta in run_metas:
        metrics_path = run_dir / run_meta["folder"] / "metrics_summary.json"
        if not metrics_path.exists():
            missing.append(str(metrics_path))
    return missing


def load_run(
    run_meta: dict,
    run_dir: Path,
) -> ParsedRun:
    metrics_path = run_dir / run_meta["folder"] / "metrics_summary.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    training_loss_path = run_dir / run_meta["folder"] / "training_loss.txt"
    summary = load_summary(metrics_path)
    config = extract_run_config_from_summary(
        summary,
        run_meta,
        metrics_path,
        training_loss_path if training_loss_path.exists() else None,
    )
    return ParsedRun(config=config, by_lang=extract_proper_term_metrics(summary))


def load_all_runs(run_metas: list[dict], run_dir: Path) -> list[ParsedRun]:
    missing = validate_metrics_paths(run_metas, run_dir)
    if missing:
        raise FileNotFoundError(
            "Missing metrics_summary.json for registered runs:\n  "
            + "\n  ".join(missing)
        )
    return [load_run(run_meta, run_dir) for run_meta in run_metas]
