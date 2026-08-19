"""Split test sentences into overlap/no_overlap subsets by training-set containment.

For each language pair, any test row whose English source sentence has at least
50% token containment overlap with any training source sentence is an "overlap"
row; otherwise it's "no_overlap". To keep the leakage-honesty comparison fair
across language pairs, every language and category is then truncated to the
same size: the minimum row count found across all three languages' overlap AND
no_overlap sets (computed at runtime, not hardcoded), keeping the first N rows
in original file order (deterministic, no random sampling).

Writes ``{output_dir}/no_overlap/{test_file}`` and ``{output_dir}/overlap/{test_file}``
for each language pair. Original test files are not modified.

Usage:
    python experiments/lora_finetuning/scripts/filter_test_sentence_overlap.py
    python experiments/lora_finetuning/scripts/filter_test_sentence_overlap.py --dry-run
    python experiments/lora_finetuning/scripts/filter_test_sentence_overlap.py \
        --training-dir path/to/training --test-dir path/to/test \
        --output-dir path/to/test_cleaned_by_sentences
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
DEFAULT_OUTPUT_DIR = SCRIPT_DIR.parent / "data" / "test_cleaned_by_sentences"

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


def compute_pair(
    pair: str,
    training_dir: Path,
    test_dir: Path,
) -> tuple[list[dict], list[dict]] | None:
    """Return (no_overlap_rows, overlap_rows) for one language pair, or None if inputs are missing."""
    training_file, test_file = PAIR_FILES[pair]
    training_path = training_dir / training_file
    test_path = test_dir / test_file

    if not training_path.exists():
        print(f"  [{pair}] WARNING: training file not found: {training_path}")
        return None
    if not test_path.exists():
        print(f"  [{pair}] WARNING: test file not found: {test_path}")
        return None

    train_token_sets = build_training_token_sets(training_path)
    test_rows = load_jsonl(test_path)

    no_overlap_rows: list[dict] = []
    overlap_rows: list[dict] = []
    empty_source_count = 0

    for row in test_rows:
        test_tokens = tokenize(source_sentence(row))
        if not test_tokens:
            empty_source_count += 1
            no_overlap_rows.append(row)
            continue
        if has_sentence_overlap(test_tokens, train_token_sets):
            overlap_rows.append(row)
        else:
            no_overlap_rows.append(row)

    print(
        f"  [{pair}] raw: {len(no_overlap_rows)} no_overlap, {len(overlap_rows)} overlap"
        f" (of {len(test_rows)})"
    )
    if empty_source_count:
        print(f"  [{pair}] {empty_source_count} row(s) with empty source counted as no_overlap")

    return no_overlap_rows, overlap_rows


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Split test sentences into overlap/no_overlap subsets by >=50% token "
            "containment with training data, balanced to the same size across "
            "all three language pairs."
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
        help=(
            "Directory to write no_overlap/ and overlap/ subdirectories into "
            "(default: data/test_cleaned_by_sentences)."
        ),
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

    per_pair: dict[str, tuple[list[dict], list[dict]]] = {}
    for pair in PAIR_FILES:
        result = compute_pair(pair, args.training_dir, args.test_dir)
        if result is not None:
            per_pair[pair] = result

    if not per_pair:
        print("\nNo language pairs computed; nothing to do.")
        return

    target_n = min(
        len(rows) for no_overlap_rows, overlap_rows in per_pair.values() for rows in (no_overlap_rows, overlap_rows)
    )
    print(f"\nBalancing every language/category to the minimum count: {target_n}")

    for pair, (no_overlap_rows, overlap_rows) in per_pair.items():
        _, test_file = PAIR_FILES[pair]
        no_overlap_balanced = no_overlap_rows[:target_n]
        overlap_balanced = overlap_rows[:target_n]

        if args.dry_run:
            print(
                f"  [{pair}] DRY-RUN - would write {len(no_overlap_balanced)} no_overlap, "
                f"{len(overlap_balanced)} overlap"
            )
        else:
            no_overlap_path = args.output_dir / "no_overlap" / test_file
            overlap_path = args.output_dir / "overlap" / test_file
            write_jsonl(no_overlap_path, no_overlap_balanced)
            write_jsonl(overlap_path, overlap_balanced)
            print(
                f"  [{pair}] wrote {len(no_overlap_balanced)} no_overlap -> {no_overlap_path}, "
                f"{len(overlap_balanced)} overlap -> {overlap_path}"
            )

    print()
    print("Done.")


if __name__ == "__main__":
    main()
