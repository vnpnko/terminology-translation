#!/usr/bin/env python3
"""
Collect English proper terms from JSONL dev files.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = sorted(
    REPO_ROOT / "data" / "sap_dev" / name
    for name in ("ende_dev_v1.jsonl", "enes_dev_v1.jsonl", "enru_dev_v1.jsonl")
)
DEFAULT_OUTPUT = REPO_ROOT / "data" / "term_pairs" / "all_eng_proper_terms.txt"


def collect_proper_terms(jsonl_path: Path) -> Counter[str]:
    counts: Counter[str] = Counter()
    with jsonl_path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            proper = record.get("proper_terms")
            for key in proper:
                counts[key] += 1
    print(f"Collected {len(counts)} proper terms from {jsonl_path}")
    return counts


def merge_case_variants(counts: Counter[str]) -> dict[str, str]:
    by_fold: dict[str, Counter[str]] = {}
    for term, n in counts.items():
        by_fold.setdefault(term.casefold(), Counter())[term] += n

    canonical: dict[str, str] = {}
    for fold, variants in by_fold.items():
        canonical[fold] = variants.most_common(1)[0][0]
    return canonical


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="JSONL file(s) to read (default: all orig/extra *_dev.jsonl)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output text file (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    args = parser.parse_args()

    total_counts: Counter[str] = Counter()
    for path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
        total_counts += collect_proper_terms(path)

    canonical = merge_case_variants(total_counts)
    sorted_terms = sorted(canonical.values(), key=str.casefold)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(sorted_terms) + ("\n" if sorted_terms else ""),
        encoding="utf-8",
    )
    print(f"Wrote {len(sorted_terms)} terms to {args.output}")


if __name__ == "__main__":
    main()
