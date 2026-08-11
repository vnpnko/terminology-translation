#!/usr/bin/env python3
"""
Check a JSONL file for duplicate source sentences within the same file.

Compares the ``en`` field only (whitespace-stripped). Does not compare across
files. Report-only — no output file is written.

Usage::

    python check_duplicate_sources.py data/processed/dev_v2/ende_dev_v2.jsonl
    python check_duplicate_sources.py path/to/file.jsonl --source-field en
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from term_utils import repo_rel_path


def find_duplicate_sources(
    path: Path,
    *,
    source_field: str = "en",
) -> dict[str, Any]:
    occurrences: dict[str, list[int]] = defaultdict(list)
    stats = {
        "total_lines": 0,
        "skipped_empty": 0,
        "skipped_malformed": 0,
        "skipped_missing_source": 0,
    }

    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                stats["skipped_empty"] += 1
                continue
            stats["total_lines"] += 1
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                stats["skipped_malformed"] += 1
                print(f"Warning: skipping malformed line {line_no}: {exc}", file=sys.stderr)
                continue
            if not isinstance(record, dict):
                stats["skipped_malformed"] += 1
                print(f"Warning: skipping non-object line {line_no}", file=sys.stderr)
                continue
            source = record.get(source_field)
            if not isinstance(source, str):
                stats["skipped_missing_source"] += 1
                print(
                    f"Warning: skipping line {line_no} (missing {source_field!r})",
                    file=sys.stderr,
                )
                continue
            key = source.strip()
            occurrences[key].append(line_no)

    duplicate_groups = [
        {"source": source, "line_numbers": line_numbers, "count": len(line_numbers)}
        for source, line_numbers in occurrences.items()
        if len(line_numbers) > 1
    ]
    duplicate_groups.sort(key=lambda g: (-g["count"], g["line_numbers"][0]))

    unique_sources = len(occurrences)
    extra_duplicate_lines = sum(g["count"] - 1 for g in duplicate_groups)

    return {
        "input_file": repo_rel_path(path),
        "source_field": source_field,
        **stats,
        "unique_sources": unique_sources,
        "duplicate_groups": len(duplicate_groups),
        "extra_duplicate_lines": extra_duplicate_lines,
        "duplicates": duplicate_groups,
    }


def print_report(report: dict[str, Any]) -> None:
    print(f"File: {report['input_file']}")
    print(f"Source field: {report['source_field']} (whitespace-stripped)")
    print(f"Total lines: {report['total_lines']}")
    print(f"Unique sources: {report['unique_sources']}")
    print(f"Duplicate groups: {report['duplicate_groups']}")
    print(f"Extra duplicate lines: {report['extra_duplicate_lines']}")

    skipped = report["skipped_empty"] + report["skipped_malformed"] + report["skipped_missing_source"]
    if skipped:
        print(
            f"Skipped: {skipped} "
            f"(empty={report['skipped_empty']}, "
            f"malformed={report['skipped_malformed']}, "
            f"missing source={report['skipped_missing_source']})"
        )

    duplicates: list[dict[str, Any]] = report["duplicates"]
    if not duplicates:
        print("\nNo duplicate source sentences found.")
        return

    print(f"\nDuplicate source sentences ({len(duplicates)}):")
    for idx, group in enumerate(duplicates, start=1):
        line_nums = ", ".join(str(n) for n in group["line_numbers"])
        print(f"\n[{idx}] lines {line_nums} ({group['count']}x)")
        print(group["source"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check a JSONL file for duplicate source sentences within the same file."
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the JSONL file to check",
    )
    parser.add_argument(
        "--source-field",
        default="en",
        help="JSON field to treat as the source sentence (default: en)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    report = find_duplicate_sources(input_path, source_field=args.source_field)
    print_report(report)

    if report["duplicate_groups"] > 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
