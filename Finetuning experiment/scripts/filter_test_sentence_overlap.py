"""Remove test sentences whose source text overlaps with training source sentences.

For each language pair, any test row whose English source sentence has at least
50% token containment overlap with any training source sentence is removed.
Kept rows are written to data/test_cleaned_gpt/; removed rows go to a companion
*.removed.jsonl file for inspection. Original test files are not modified.

Usage:
    python "Finetuning experiment/scripts/filter_test_sentence_overlap.py"
    python "Finetuning experiment/scripts/filter_test_sentence_overlap.py" --dry-run
    python "Finetuning experiment/scripts/filter_test_sentence_overlap.py" \
        --training-dir path/to/training --test-dir path/to/test \
        --output-dir path/to/test_cleaned_gpt
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
DEFAULT_TEST_DIR = SCRIPT_DIR.parent / "data" / "test"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "test_cleaned_gpt"

OVERLAP_THRESHOLD = 0.5

# Maps language-pair prefix -> (training filename, test filename)
PAIR_FILES: dict[str, tuple[str, str]] = {
    "ende": ("ende_dev_v2_training.jsonl", "ende_dev_v1_test.jsonl"),
    "enes": ("enes_dev_v2_training.jsonl", "enes_dev_v1_test.jsonl"),
    "enru": ("enru_dev_v2_training.jsonl", "enru_dev_v1_test.jsonl"),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def normalize(text: str) -> str:
    """Lowercase and collapse internal whitespace."""
    return " ".join(text.split()).lower()


def tokenize(text: str) -> set[str]:
    """Return whitespace tokens from normalized text."""
    return set(normalize(text).split())


def source_sentence(record: dict) -> str:
    return record.get("en", "")


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def containment_ratio(test_tokens: set[str], train_tokens: set[str]) -> float:
    if not test_tokens:
        return 0.0
    return len(test_tokens & train_tokens) / len(test_tokens)


def has_sentence_overlap(test_tokens: set[str], train_token_sets: list[set[str]]) -> bool:
    return any(
        containment_ratio(test_tokens, train_tokens) >= OVERLAP_THRESHOLD
        for train_tokens in train_token_sets
    )


def build_training_token_sets(training_path: Path) -> list[set[str]]:
    return [tokenize(source_sentence(record)) for record in load_jsonl(training_path)]


def filter_pair(
    pair: str,
    training_dir: Path,
    test_dir: Path,
    output_dir: Path,
    dry_run: bool,
) -> None:
    training_file, test_file = PAIR_FILES[pair]
    training_path = training_dir / training_file
    test_path = test_dir / test_file
    output_path = output_dir / test_file
    removed_path = output_dir / test_file.replace(".jsonl", ".removed.jsonl")

    if not training_path.exists():
        print(f"  [{pair}] WARNING: training file not found: {training_path}")
        return
    if not test_path.exists():
        print(f"  [{pair}] WARNING: test file not found: {test_path}")
        return

    train_token_sets = build_training_token_sets(training_path)
    test_rows = load_jsonl(test_path)

    kept: list[dict] = []
    removed: list[dict] = []
    empty_source_count = 0

    for row in test_rows:
        test_tokens = tokenize(source_sentence(row))
        if not test_tokens:
            empty_source_count += 1
            kept.append(row)
            continue
        if has_sentence_overlap(test_tokens, train_token_sets):
            removed.append(row)
        else:
            kept.append(row)

    total = len(test_rows)
    n_kept = len(kept)
    n_removed = len(removed)

    if dry_run:
        print(
            f"  [{pair}] DRY-RUN - would keep {n_kept}, remove {n_removed} (of {total})"
        )
        if empty_source_count:
            print(f"  [{pair}] DRY-RUN - {empty_source_count} row(s) with empty source kept")
    else:
        write_jsonl(output_path, kept)
        write_jsonl(removed_path, removed)
        print(
            f"  [{pair}] kept {n_kept}, removed {n_removed} (of {total})"
            f" -> {output_path.name}, removed rows in {removed_path.name}"
        )
        if empty_source_count:
            print(f"  [{pair}] {empty_source_count} row(s) with empty source kept")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove test sentences whose source text has >=50% token containment "
            "overlap with any training source sentence."
        )
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
        help="Directory containing test JSONL files (default: data/test).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for filtered output files (default: data/test_cleaned_gpt).",
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
    print(f"filter_test_sentence_overlap - {mode_label}")
    print(f"  training dir : {args.training_dir}")
    print(f"  test dir     : {args.test_dir}")
    print(f"  output dir   : {args.output_dir}")
    print()

    for pair in PAIR_FILES:
        filter_pair(
            pair,
            args.training_dir,
            args.test_dir,
            args.output_dir,
            args.dry_run,
        )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
