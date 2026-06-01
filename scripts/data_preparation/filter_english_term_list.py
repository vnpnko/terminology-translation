#!/usr/bin/env python3
"""
Filter collected English proper terms to keep likely domain terminology.

The JSONL ``proper_terms`` keys include many common English words (e.g. "about",
"type", "run") that are not fixed terminology. This script drops those using:

  1. English stopwords (single-token terms only)
  2. A curated blocklist (``scripts/data/proper_term_blocklist.txt``)
  3. Optional Zipf frequency threshold via ``wordfreq`` (``--max-zipf``)

Multi-word phrases are kept by default (e.g. "access control"), except entries
on the blocklist (e.g. "change document").

Usage::

    python filter_proper_terms.py
    python filter_proper_terms.py --max-zipf 5.2
    python filter_proper_terms.py --blocklist path/to/extra_blocklist.txt
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = REPO_ROOT / "data" / "term_pairs" / "all_eng_proper_terms.txt"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "term_pairs" / "filtered_eng_proper_terms.txt"
DEFAULT_BLOCKLIST = SCRIPTS_DIR / "data" / "proper_term_blocklist.txt"
DEFAULT_ALLOWLIST = SCRIPTS_DIR / "data" / "proper_term_allowlist.txt"

# Minimal stopword set (single-token filter only).
STOPWORDS = frozenset(
    """
    a about above across after again against all almost alone along already also
    although always am among an and another any anybody anyone anything anyway
    anywhere are around as at be became because become becomes been before behind
    being below between beyond both but by can cannot could did do does doing done
    down during each either else enough even ever every for from further had has
    have having he her here him his how however i if in into is it its itself
    just me more most much must my myself never no nor not now of off on once one
    only or other our out over own same she should since so some such than that
    the their them then there these they this those though through to too under
    until up us very was we well were what when where which while who will with
    without would you your
    """.split()
)


def load_term_set(path: Path) -> frozenset[str]:
    """Load lowercased terms from a text file (``#`` comments, blank lines skipped)."""
    if not path.is_file():
        raise FileNotFoundError(path)
    terms: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            terms.add(line.casefold())
    return frozenset(terms)


def merge_case_variants(terms: list[str]) -> list[str]:
    """Pick one spelling per case-insensitive form (prefer most common casing)."""
    by_fold: dict[str, Counter[str]] = {}
    for term in terms:
        by_fold.setdefault(term.casefold(), Counter())[term] += 1
    return [variants.most_common(1)[0][0] for variants in by_fold.values()]


def zipf_frequency(word: str) -> float:
    import wordfreq

    return wordfreq.zipf_frequency(word.lower(), "en")


def is_proper_term(
    term: str,
    *,
    blocklist: frozenset[str],
    allowlist: frozenset[str],
    max_zipf: float | None,
) -> bool:
    term = term.strip()
    if not term:
        return False

    folded = term.casefold()
    if folded in blocklist:
        return False

    tokens = term.split()
    if len(tokens) == 1:
        word = tokens[0].lower()
        if word in STOPWORDS:
            return False
        if max_zipf is not None and folded not in allowlist:
            try:
                if zipf_frequency(word) >= max_zipf:
                    return False
            except ImportError as exc:
                raise ImportError(
                    "Install wordfreq for --max-zipf: pip install wordfreq"
                ) from exc
        return True

    # Multi-word: drop only if explicitly blocklisted (already checked).
    if max_zipf is not None:
        non_stop = [t.lower() for t in tokens if t.lower() not in STOPWORDS]
        if non_stop and all(t.casefold() not in allowlist for t in non_stop):
            try:
                if all(zipf_frequency(t) >= max_zipf for t in non_stop):
                    return False
            except ImportError as exc:
                raise ImportError(
                    "Install wordfreq for --max-zipf: pip install wordfreq"
                ) from exc
    return True


def filter_terms(
    terms: list[str],
    *,
    blocklist: frozenset[str],
    allowlist: frozenset[str],
    max_zipf: float | None,
) -> list[str]:
    canonical = merge_case_variants(terms)
    kept = [
        t
        for t in canonical
        if is_proper_term(
            t, blocklist=blocklist, allowlist=allowlist, max_zipf=max_zipf
        )
    ]
    return sorted(kept, key=str.casefold)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Filter collected English proper terms to likely terminology."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input term list (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output term list (default: {DEFAULT_OUTPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "--blocklist",
        type=Path,
        action="append",
        default=[DEFAULT_BLOCKLIST],
        help="Blocklist file(s); terms matched case-insensitively (repeatable)",
    )
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=DEFAULT_ALLOWLIST,
        help=f"Allowlist for --max-zipf (default: {DEFAULT_ALLOWLIST.name})",
    )
    parser.add_argument(
        "--max-zipf",
        type=float,
        default=None,
        metavar="SCORE",
        help=(
            "Also drop single words with Zipf frequency >= SCORE (needs wordfreq). "
            "Multi-word phrases are dropped only when every non-stopword token "
            "meets the threshold and none are allowlisted. Try 5.2–5.3."
        ),
    )
    parser.add_argument(
        "--report-removed",
        action="store_true",
        help="Print removed terms to stderr",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(args.input)

    blocklist: set[str] = set()
    for path in args.blocklist:
        blocklist |= set(load_term_set(path))
    allowlist = load_term_set(args.allowlist)

    raw = [
        line.strip()
        for line in args.input.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    canonical = merge_case_variants(raw)
    kept = filter_terms(
        raw,
        blocklist=frozenset(blocklist),
        allowlist=allowlist,
        max_zipf=args.max_zipf,
    )
    removed = sorted(
        (t for t in canonical if t not in set(kept)), key=str.casefold
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(kept) + ("\n" if kept else ""),
        encoding="utf-8",
    )

    print(
        f"Read {len(raw)} lines ({len(canonical)} unique) from {args.input}\n"
        f"Removed {len(removed)} terms, wrote {len(kept)} to {args.output}"
    )
    if args.report_removed and removed:
        print("\nRemoved terms:", file=__import__("sys").stderr)
        for term in removed:
            print(f"  {term}", file=__import__("sys").stderr)


if __name__ == "__main__":
    main()
