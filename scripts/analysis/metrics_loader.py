"""Shared metrics_summary.json loading for analysis scripts and poster figures."""

from __future__ import annotations

import json
import math
from pathlib import Path

LANG_ORDER = ("ende", "enru", "enes")
LANG_LABELS = {
    "ende": "EN→DE",
    "enru": "EN→RU",
    "enes": "EN→ES",
}
MODE_ORDER = ("no_term", "proper_term", "random_term")
DEFAULT_MODE = "proper_term"

BASELINE_DIRS = ("gpt", "qwen_3b", "qwen_7b")
BASELINE_LABELS = {
    "gpt": "GPT",
    "qwen_3b": "Qwen 3B",
    "qwen_7b": "Qwen 7B",
}

METRIC_SPECS: dict[str, str | tuple[str, str]] = {
    "bleu": "bleu",
    "chrf": "chrf",
    "term_accuracy_pct": ("terminology_accuracy", "avg_ratio_pct"),
    "macro_avg_consistency": ("terminology_consistency", "macro_avg_consistency"),
    "weighted_avg_consistency": ("terminology_consistency", "weighted_avg_consistency"),
    "total_terms": ("terminology_accuracy", "total_terms"),
}

METRICS = tuple(
    (key, METRIC_SPECS[key], label)
    for key, label in (
        ("bleu", "BLEU"),
        ("chrf", "chrF"),
        ("term_accuracy_pct", "Term Accuracy %"),
        ("macro_avg_consistency", "Macro Consistency"),
        ("weighted_avg_consistency", "Weighted Consistency"),
    )
)


def load_summary(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def extract_metric(metrics: dict, spec: str | tuple[str, str]) -> float | int | None:
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


def extract_mode_metrics(summary: dict) -> dict[tuple[str, str], dict[str, float | int | None]]:
    by_lang_mode: dict[tuple[str, str], dict[str, float | int | None]] = {}

    for lang in LANG_ORDER:
        lang_data = summary.get("languages", {}).get(lang)
        if not lang_data:
            continue

        for mode in MODE_ORDER:
            mode_data = lang_data.get("modes", {}).get(mode)
            if not mode_data:
                continue

            metrics = mode_data.get("metrics", {})
            by_lang_mode[(lang, mode)] = {
                column: extract_metric(metrics, spec) for column, spec in METRIC_SPECS.items()
            }

    return by_lang_mode


def get_lang_mode_metrics(
    summary: dict,
    lang: str,
    mode: str = DEFAULT_MODE,
) -> dict[str, float | int | None]:
    lang_data = summary.get("languages", {}).get(lang, {})
    mode_data = lang_data.get("modes", {}).get(mode, {})
    metrics = mode_data.get("metrics", {})
    return {column: extract_metric(metrics, spec) for column, spec in METRIC_SPECS.items()}


def macro_average(
    summary: dict,
    mode: str = DEFAULT_MODE,
    metric_keys: tuple[str, ...] = ("bleu", "chrf", "term_accuracy_pct", "macro_avg_consistency", "weighted_avg_consistency"),
) -> dict[str, float | int | None]:
    values: dict[str, list[float | int]] = {key: [] for key in metric_keys}
    total_terms = 0
    has_terms = False

    for lang in LANG_ORDER:
        row = get_lang_mode_metrics(summary, lang, mode)
        for key in metric_keys:
            val = row.get(key)
            if val is not None:
                values[key].append(val)
        terms = row.get("total_terms")
        if terms is not None:
            total_terms += int(terms)
            has_terms = True

    result: dict[str, float | int | None] = {}
    for key in metric_keys:
        vals = values[key]
        result[key] = sum(vals) / len(vals) if vals else None
    result["total_terms"] = total_terms if has_terms else None
    return result


def load_metrics_path(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics file: {path}")
    return load_summary(path)


def require_paths(paths: list[Path]) -> None:
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing metrics files:\n  " + "\n  ".join(missing))
