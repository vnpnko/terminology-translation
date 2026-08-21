"""Remove dev_v2 lines whose English source also appears in dev_v1/original.

Writes filtered files to
``experiments/lora_finetuning/shared/data/dev_v2_deduped/`` — never overwrites
the ``data/dev_v2/`` or ``data/dev_v1/dev_v1_original/``
inputs. Reads dev_v2 sources from ``data/dev_v2/`` and
filters out any line whose ``en`` text also appears in
``data/dev_v1/dev_v1_original/``.

Usage::

    python remove_dev_v2_overlap.py --all
    python remove_dev_v2_overlap.py -t de
    python remove_dev_v2_overlap.py --all --force
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[4]

DEV_V2_DIR = PROJECT_ROOT / "shared" / "data" / "dev_v2"
DEV_V1_ORIGINAL_DIR = PROJECT_ROOT / "shared" / "data" / "dev_v1" / "dev_v1_original"
DEV_V2_DEDUPED_DIR = PROJECT_ROOT / "experiments" / "lora_finetuning" / "shared" / "data" / "dev_v2_deduped"


@dataclass(frozen=True)
class LangPair:
    prefix: str
    input_name: str


LANG_PAIRS: dict[str, LangPair] = {
    "ende": LangPair("ende", "ende_dev_v2.jsonl"),
    "enes": LangPair("enes", "enes_dev_v2.jsonl"),
    "enru": LangPair("enru", "enru_dev_v2.jsonl"),
}


def repo_rel_path(path: Path) -> str:
    """Return a repo-root-relative path string for portable JSON reports."""
    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def refuse_if_exists(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"Output already exists: {path}\n"
            "Refusing to overwrite. Pass --force to regenerate."
        )


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dev_v1_input_name(lang_pair: LangPair) -> str:
    return f"{lang_pair.prefix}_dev_v1.jsonl"


def load_filter_en_set(path: Path) -> set[str]:
    en_set: set[str] = set()
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                print(f"Warning: skipping malformed filter line {line_no} in {path}: {exc}")
                continue
            if not isinstance(record, dict):
                print(f"Warning: skipping non-object filter line {line_no} in {path}")
                continue
            en = record.get("en")
            if not isinstance(en, str):
                print(f"Warning: skipping filter line {line_no} in {path} (missing en)")
                continue
            en_set.add(en.strip())
    return en_set


def filter_dev_v2(
    input_path: Path,
    filter_en: set[str],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    stats = {
        "input_lines": 0,
        "removed": 0,
        "kept": 0,
        "skipped_malformed": 0,
    }

    with input_path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            stats["input_lines"] += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                stats["skipped_malformed"] += 1
                print(f"Warning: skipping malformed source line {line_no} in {input_path}: {exc}")
                continue
            if not isinstance(record, dict):
                stats["skipped_malformed"] += 1
                print(f"Warning: skipping non-object source line {line_no} in {input_path}")
                continue
            en = record.get("en")
            if not isinstance(en, str):
                stats["skipped_malformed"] += 1
                print(f"Warning: skipping source line {line_no} in {input_path} (missing en)")
                continue
            if en.strip() in filter_en:
                stats["removed"] += 1
                continue
            kept.append(record)
            stats["kept"] += 1

    return kept, stats


def remove_overlap_for_pair(
    lang_pair: LangPair,
    *,
    force: bool,
) -> dict[str, Any]:
    filter_path = DEV_V1_ORIGINAL_DIR / dev_v1_input_name(lang_pair)
    input_path = DEV_V2_DIR / lang_pair.input_name
    output_path = DEV_V2_DEDUPED_DIR / lang_pair.input_name

    if not filter_path.is_file():
        raise SystemExit(f"Filter file not found: {filter_path}")
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    refuse_if_exists(output_path, force=force)

    filter_en = load_filter_en_set(filter_path)
    kept, stats = filter_dev_v2(input_path, filter_en)
    save_jsonl(output_path, kept)

    report = {
        "lang_pair": lang_pair.prefix,
        "filter_file": repo_rel_path(filter_path),
        "input_file": repo_rel_path(input_path),
        "output_file": repo_rel_path(output_path),
        "filter_unique_en": len(filter_en),
        **stats,
    }
    print(
        f"{lang_pair.prefix}: {stats['input_lines']} in, "
        f"{stats['removed']} removed, {stats['kept']} kept"
        + (f", {stats['skipped_malformed']} skipped" if stats["skipped_malformed"] else "")
        + f" -> {output_path}"
    )
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove dev_v2 lines whose en text appears in dev_v1/original."
    )
    parser.add_argument("-t", "--target-lang", choices=["de", "es", "ru"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output in experiments/lora_finetuning/shared/data/dev_v2_deduped/",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and args.target_lang is None:
        raise SystemExit("Specify --all or -t de|es|ru")

    target_to_prefix = {"de": "ende", "es": "enes", "ru": "enru"}
    if args.all:
        pairs = [LANG_PAIRS["ende"], LANG_PAIRS["enes"], LANG_PAIRS["enru"]]
    else:
        pairs = [LANG_PAIRS[target_to_prefix[args.target_lang]]]

    reports: dict[str, Any] = {}
    for lang_pair in pairs:
        reports[lang_pair.prefix] = remove_overlap_for_pair(lang_pair, force=args.force)

    totals = {
        "input_lines": sum(r["input_lines"] for r in reports.values()),
        "removed": sum(r["removed"] for r in reports.values()),
        "kept": sum(r["kept"] for r in reports.values()),
        "skipped_malformed": sum(r["skipped_malformed"] for r in reports.values()),
    }
    print(
        f"\nTotal: {totals['input_lines']} in, "
        f"{totals['removed']} removed, {totals['kept']} kept"
        + (f", {totals['skipped_malformed']} skipped" if totals["skipped_malformed"] else "")
    )

    report_path = DEV_V2_DEDUPED_DIR / "remove_overlap_report.json"
    save_json(report_path, {"pairs": reports, "totals": totals})
    print(f"Wrote report to {report_path}")


if __name__ == "__main__":
    main()
