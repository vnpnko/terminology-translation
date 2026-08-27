"""Shared data-loading and constants for the by_language_pair/gpt and
by_language_pair/qwen figure scripts -- both compare term-list variants
by language pair, differing only in which baseline model(s) they plot.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))

from shared.lib.analysis.metrics_loader import (
    BASELINE_DIRS,
    DEFAULT_MODE,
    LANG_ORDER,
    NO_TERM_MODE,
    RANDOM_TERM_MODE,
    get_lang_mode_metrics,
    load_metrics_path,
    require_paths,
)

NO_TERM_KEY = "no_term"
RANDOM_TERM_KEY = "random_term"
BASELINE_SOURCE_STRATEGY = "original"
NO_TERM_LABEL = "No term"
RANDOM_TERM_LABEL = "Random term"

# "dictionary" reads shared/results/dev_v1/dictionary/ (the real term-list
# variant built by experiments/term_expansion/dictionary/'s
# build_term_dictionary.py/apply_dictionary_to_dev_v1.py) via the same
# _strategy_dir() fallback used for expand/cleaned below -- not the unrelated
# dev_v2-on-its-own-terms comparison the old EXTERNAL_DICTIONARY_* constants
# pointed at.
STRATEGY_ORDER = ("original", "expand", "cleaned", "dictionary")
SERIES_ORDER = (NO_TERM_KEY, RANDOM_TERM_KEY, *STRATEGY_ORDER)
STRATEGY_LABELS = {
    NO_TERM_KEY: NO_TERM_LABEL,
    RANDOM_TERM_KEY: RANDOM_TERM_LABEL,
    "original": "Original",
    "expand": "GPT-expanded",
    "cleaned": "GPT-cleaned",
    "dictionary": "Dictionary",
}


def _strategy_dir(results_root: Path, strategy: str) -> Path:
    if strategy == BASELINE_SOURCE_STRATEGY:
        return results_root / "dev_v1" / strategy / "few_shot"
    return results_root / "dev_v1" / strategy


def collect_data(
    results_root: Path,
) -> dict[str, dict[str, dict[str, dict[str, float | None]]]]:
    paths = [
        _strategy_dir(results_root, strategy) / baseline / "metrics_summary.json"
        for strategy in STRATEGY_ORDER
        for baseline in BASELINE_DIRS
    ]
    require_paths(paths)

    data: dict[str, dict[str, dict[str, dict[str, float | None]]]] = {
        baseline: {} for baseline in BASELINE_DIRS
    }
    for strategy in STRATEGY_ORDER:
        for baseline in BASELINE_DIRS:
            summary = load_metrics_path(
                _strategy_dir(results_root, strategy) / baseline / "metrics_summary.json"
            )
            data[baseline][strategy] = {
                lang: get_lang_mode_metrics(summary, lang, DEFAULT_MODE) for lang in LANG_ORDER
            }
            if strategy == BASELINE_SOURCE_STRATEGY:
                data[baseline][NO_TERM_KEY] = {
                    lang: get_lang_mode_metrics(summary, lang, NO_TERM_MODE)
                    for lang in LANG_ORDER
                }
                data[baseline][RANDOM_TERM_KEY] = {
                    lang: get_lang_mode_metrics(summary, lang, RANDOM_TERM_MODE)
                    for lang in LANG_ORDER
                }

    return data
