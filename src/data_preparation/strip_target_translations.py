"""Strip non-English translations from JSONL files in extra/.

Writes back in place (with ``--dry-run`` to preview first). Keeps the
English sentence and term keys; clears translation sentences (de, es, ru,
...) and translation values in proper_terms / random_terms. Input defaults
to ``data/extra/*_dev.jsonl`` — no longer present in this repo after the
open-sourcing restructure; pass ``--input`` to target real files.

Usage::

    python strip_target_translations.py
    python strip_target_translations.py --input data/extra/ende_dev.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = sorted((REPO_ROOT / "data" / "extra").glob("*_dev.jsonl"))
TERM_FIELDS = frozenset({"proper_terms", "random_terms"})


def strip_record(record: dict[str, Any]) -> dict[str, Any]:
    out: dict = {}
    for key, value in record.items():
        if key == "en":
            out[key] = value
        elif key in TERM_FIELDS:
            out[key] = {term: "" for term in value}
        else:
            out[key] = ""
    return out


def process_file(path: Path, *, dry_run: bool = False) -> int:
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.rstrip("\n")
            if not raw.strip():
                lines.append(line if line.endswith("\n") else line + "\n")
                continue
            record = json.loads(raw)
            lines.append(json.dumps(strip_record(record), ensure_ascii=False) + "\n")

    if dry_run:
        print(f"Would update {len(lines)} records in {path}")
        return len(lines)

    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(path)
    print(f"Updated {len(lines)} records in {path}")
    return len(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clear translation text in extra/ JSONL dev files."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="JSONL file(s) to process (default: extra/*_dev.jsonl)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be changed without writing files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input:
        raise SystemExit("No input files matched.")

    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
        process_file(path, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
