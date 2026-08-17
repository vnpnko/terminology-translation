"""Remove test sentences whose source-side terms overlap with training source-side terms.

For each language pair, any sentence in data/test_cleaned whose source-term keys
(from both proper_terms and random_terms) match — case-insensitively, after whitespace
normalization — at least one source-term key from the corresponding training file is
removed.  The filtered file overwrites the original; removed rows are written to a
companion *.removed.jsonl file for inspection.

Usage:
    python experiments/04_lora_finetuning/scripts/filter_test_term_overlap.py
    python experiments/04_lora_finetuning/scripts/filter_test_term_overlap.py --dry-run
    python experiments/04_lora_finetuning/scripts/filter_test_term_overlap.py \
        --training-dir path/to/training --test-dir path/to/test_cleaned
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_TRAINING_DIR = SCRIPT_DIR.parent / "data" / "training"
DEFAULT_TEST_DIR = SCRIPT_DIR.parent / "data" / "test_cleaned"

# Maps language-pair prefix -> (training filename, test filename)
PAIR_FILES: dict[str, tuple[str, str]] = {
    "ende": ("ende_dev_v2_training.jsonl", "ende_dev_v1_test.jsonl"),
    "enes": ("enes_dev_v2_training.jsonl", "enes_dev_v1_test.jsonl"),
    "enru": ("enru_dev_v2_training.jsonl", "enru_dev_v1_test.jsonl"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(term: str) -> str:
    """Lowercase and collapse internal whitespace."""
    return " ".join(term.split()).lower()


def source_terms(record: dict) -> set[str]:
    """Return the normalized source-side term keys from both term dicts."""
    terms: set[str] = set()
    for field in ("proper_terms", "random_terms"):
        for key in record.get(field, {}).keys():
            terms.add(normalize(key))
    return terms


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def build_training_term_set(training_path: Path) -> set[str]:
    """Collect all normalized source-side term keys from a training file."""
    term_set: set[str] = set()
    for record in load_jsonl(training_path):
        term_set.update(source_terms(record))
    return term_set


def filter_pair(
    pair: str,
    training_dir: Path,
    test_dir: Path,
    dry_run: bool,
) -> None:
    training_file, test_file = PAIR_FILES[pair]
    training_path = training_dir / training_file
    test_path = test_dir / test_file
    removed_path = test_dir / test_file.replace(".jsonl", ".removed.jsonl")

    if not training_path.exists():
        print(f"  [{pair}] WARNING: training file not found: {training_path}")
        return
    if not test_path.exists():
        print(f"  [{pair}] WARNING: test file not found: {test_path}")
        return

    train_terms = build_training_term_set(training_path)
    test_rows = load_jsonl(test_path)

    kept: list[dict] = []
    removed: list[dict] = []
    for row in test_rows:
        row_terms = source_terms(row)
        if row_terms.isdisjoint(train_terms):
            kept.append(row)
        else:
            removed.append(row)

    total = len(test_rows)
    n_kept = len(kept)
    n_removed = len(removed)

    if dry_run:
        print(
            f"  [{pair}] DRY-RUN — would keep {n_kept}, remove {n_removed} (of {total})"
        )
    else:
        write_jsonl(test_path, kept)
        write_jsonl(removed_path, removed)
        print(
            f"  [{pair}] kept {n_kept}, removed {n_removed} (of {total})"
            f" → removed rows saved to {removed_path.name}"
        )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove test sentences whose source terms overlap with training."
    )
    parser.add_argument(
        "--training-dir",
        type=Path,
        default=DEFAULT_TRAINING_DIR,
        help="Directory containing training JSONL files (default: data/training).",
    )
    parser.add_argument(
        "--test-dir",
        type=Path,
        default=DEFAULT_TEST_DIR,
        help="Directory containing test_cleaned JSONL files (default: data/test_cleaned).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts only; do not write any files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode_label = "DRY-RUN" if args.dry_run else "LIVE"
    print(f"filter_test_term_overlap — {mode_label}")
    print(f"  training dir : {args.training_dir}")
    print(f"  test dir     : {args.test_dir}")
    print()

    for pair in PAIR_FILES:
        filter_pair(pair, args.training_dir, args.test_dir, args.dry_run)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
