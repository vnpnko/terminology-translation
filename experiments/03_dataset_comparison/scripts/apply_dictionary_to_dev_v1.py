"""Apply term dictionary to dev_v1 original JSONL (optional secondary step).

Writes ``{lang}_dev_v1_dictionary.jsonl`` to ``data/dev_v1/dev_v1_dictionary/``
only — never overwrites the original inputs. Reads the dictionary from
``experiments/03_dataset_comparison/data/dev_v2_dictionary/`` and enriches
``proper_terms`` in dev_v1 (``data/dev_v1/dev_v1_original/``) using
reference-based disambiguation; terms with an ambiguous dictionary
translation are silently skipped rather than added.

Usage::

    python apply_dictionary_to_dev_v1.py --all
    python apply_dictionary_to_dev_v1.py -t de --force
"""

from __future__ import annotations

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation.term_utils import (
    DEV_V1_DICTIONARY_DIR,
    DEV_V1_ORIGINAL_DIR,
    LANG_PAIRS,
    LangPair,
    TERM_DICTIONARY_DIR,
    collect_en_forms,
    collect_target_forms,
    entry_matches_reference,
    find_best_substring_match,
    lemma_key,
    load_jsonl,
    locate_substring,
    normalize_key,
    refuse_if_exists,
    save_jsonl,
)


def dev_v1_input_name(lang_pair: LangPair) -> str:
    return f"{lang_pair.prefix}_dev_v1.jsonl"


def dev_v1_output_name(lang_pair: LangPair) -> str:
    return f"{lang_pair.prefix}_dev_v1_dictionary.jsonl"


def load_dictionary(path: Path, lang_code: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise SystemExit(f"Dictionary not found: {path}")
    entries = load_jsonl(path)
    for entry in entries:
        entry["_lemma_tgt_key"] = lemma_key(
            entry.get("lemma_tgt") or entry.get("tgt_surface", ""),
            lang_code,
        )
        entry["_lemma_en_key"] = lemma_key(
            entry.get("lemma_en") or entry.get("en_surface", ""),
            "en",
        )
    return entries


def build_form_index(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Map normalized EN surface/lemma -> dictionary entries."""
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen: dict[str, set[tuple[Any, Any, Any]]] = defaultdict(set)
    for entry in entries:
        entry_key = (
            entry.get("line_id"),
            entry.get("en_surface"),
            entry.get("tgt_surface"),
        )
        for form in collect_en_forms(entry):
            norm = normalize_key(form)
            if entry_key in seen[norm]:
                continue
            seen[norm].add(entry_key)
            index[norm].append(entry)
    return index


def find_en_matches(en: str, form: str) -> list[tuple[int, int, str]]:
    """Return (start, end, matched_surface) for all case-insensitive matches."""
    matches: list[tuple[int, int, str]] = []
    pattern = re.compile(re.escape(form), flags=re.IGNORECASE)
    for match in pattern.finditer(en):
        matches.append((match.start(), match.end(), en[match.start() : match.end()]))
    return matches


def group_entries_by_translation_lemma(
    entries: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        key = entry.get("_lemma_tgt_key") or ""
        if not key:
            continue
        groups[key].append(entry)
    return groups


def resolve_target_span(
    entries: list[dict[str, Any]],
    reference: str,
) -> str | None:
    forms: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for form in collect_target_forms(entry):
            norm = normalize_key(form)
            if norm not in seen:
                seen.add(norm)
                forms.append(form)
    return find_best_substring_match(forms, reference)


def apply_dictionary_to_record(
    record: dict[str, Any],
    *,
    lang_pair: LangPair,
    form_index: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    en = str(record.get("en", ""))
    reference = str(record.get(lang_pair.tgt_code, ""))
    proper_terms = dict(record.get("proper_terms") or {})
    random_terms = record.get("random_terms") or {}
    blocked = {normalize_key(k) for k in proper_terms}
    if isinstance(random_terms, dict):
        blocked.update(normalize_key(k) for k in random_terms)

    additions: dict[str, str] = {}

    candidate_spans: list[tuple[int, int, str]] = []
    seen_span: set[tuple[int, int, str]] = set()
    for norm_form, entries in form_index.items():
        if norm_form in blocked:
            continue
        for entry in entries:
            for form in collect_en_forms(entry):
                for start, end, surface in find_en_matches(en, form):
                    key = (start, end, normalize_key(surface))
                    if key in seen_span:
                        continue
                    seen_span.add(key)
                    candidate_spans.append((start, end, surface))

    candidate_spans.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    used_spans: list[tuple[int, int]] = []
    for start, end, surface in candidate_spans:
        if any(not (end <= s or start >= e) for s, e in used_spans):
            continue

        matching_entries = [
            e
            for e in form_index.get(normalize_key(surface), [])
            if entry_matches_reference(e, reference)
            and locate_substring(surface, en) is not None
        ]
        if not matching_entries:
            continue

        lemma_groups = group_entries_by_translation_lemma(matching_entries)
        if len(lemma_groups) > 1:
            # Dictionary disagrees on the translation lemma for this term — skip
            # rather than guess.
            continue

        group_entries = next(iter(lemma_groups.values()))
        tgt_span = resolve_target_span(group_entries, reference)
        if tgt_span is None:
            continue

        src_norm = normalize_key(surface)
        if src_norm in blocked or src_norm in {normalize_key(k) for k in additions}:
            continue

        additions[surface] = tgt_span
        used_spans.append((start, end))

    out = dict(record)
    merged = dict(proper_terms)
    merged.update(additions)
    out["proper_terms"] = merged
    return out


def apply_for_pair(
    lang_pair: LangPair,
    *,
    force: bool,
) -> None:
    dict_path = TERM_DICTIONARY_DIR / lang_pair.output_name
    input_path = DEV_V1_ORIGINAL_DIR / dev_v1_input_name(lang_pair)
    output_path = DEV_V1_DICTIONARY_DIR / dev_v1_output_name(lang_pair)

    refuse_if_exists(output_path, force=force)

    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    entries = load_dictionary(dict_path, lang_pair.tgt_code)
    form_index = build_form_index(entries)
    records = load_jsonl(input_path)

    enriched: list[dict[str, Any]] = []
    terms_added_total = 0

    for record in records:
        out = apply_dictionary_to_record(
            record,
            lang_pair=lang_pair,
            form_index=form_index,
        )
        enriched.append(out)
        before = len(record.get("proper_terms") or {})
        after = len(out.get("proper_terms") or {})
        terms_added_total += after - before

    save_jsonl(output_path, enriched)
    print(f"Wrote {output_path} (+{terms_added_total} terms)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply term dictionary to dev_v1 (writes new files only)."
    )
    parser.add_argument("-t", "--target-lang", choices=["de", "es", "ru"])
    parser.add_argument("--all", action="store_true")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output in data/dev_v1/dev_v1_dictionary/",
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

    for lang_pair in pairs:
        apply_for_pair(lang_pair, force=args.force)


if __name__ == "__main__":
    main()
