#!/usr/bin/env python3
"""Build JSONL outputs from SAP_postedits term data."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Iterable, List, Tuple


def load_lines(path: Path) -> List[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_terms_line(line: str) -> List[Tuple[str, str]]:
    stripped = line.strip()
    if not stripped:
        return []
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid terms line: {stripped}") from exc
    if not isinstance(parsed, list):
        return []
    terms: List[Tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, tuple) or len(item) < 3:
            continue
        en_term, _mt_term, de_term = item[0], item[1], item[2]
        if isinstance(en_term, str) and isinstance(de_term, str):
            terms.append((en_term, de_term))
    return terms


def main() -> None:
    parser = argparse.ArgumentParser(description="Build JSONL outputs with proper terms.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "SAP_postedits",
        help="Directory containing term_postedits.test.ende.* files",
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data"
        / "SAP_postedits"
        / "term_postedits.test.ende.jsonl",
        help="Output JSONL with sentences and terms",
    )
    parser.add_argument(
        "--out-terms-jsonl",
        type=Path,
        default=Path(__file__).resolve().parent
        / "data"
        / "SAP_postedits"
        / "term_postedits.test.ende.terms.jsonl",
        help="Output JSONL with term pairs only",
    )
    args = parser.parse_args()

    src_path = args.data_dir / "term_postedits.test.ende.src"
    pe_path = args.data_dir / "term_postedits.test.ende.pe"
    mt_path = args.data_dir / "term_postedits.test.ende.mt"
    terms_path = args.data_dir / "term_postedits.test.ende.terms"

    src_lines = load_lines(src_path)
    pe_lines = load_lines(pe_path)
    mt_lines = load_lines(mt_path)
    terms_lines = load_lines(terms_path)

    if not (len(src_lines) == len(pe_lines) == len(mt_lines) == len(terms_lines)):
        raise ValueError(
            "Line counts do not match: "
            f"src={len(src_lines)}, pe={len(pe_lines)}, mt={len(mt_lines)}, terms={len(terms_lines)}"
        )

    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_terms_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with args.out_jsonl.open("w", encoding="utf-8") as out_sentences, args.out_terms_jsonl.open(
        "w", encoding="utf-8"
    ) as out_terms:
        for idx, (src, pe, mt, terms_line) in enumerate(
            zip(src_lines, pe_lines, mt_lines, terms_lines), start=1
        ):
            term_pairs = parse_terms_line(terms_line)
            proper_terms = {en: de for en, de in term_pairs}

            random_terms = {}

            sentence_obj = {
                "en": src,
                "de": pe,
                "proper_terms": proper_terms,
                "random_terms": random_terms,
            }
            out_sentences.write(json.dumps(sentence_obj, ensure_ascii=False) + "\n")

            terms_obj = {
                "row": idx,
                "terms": [{"en": en, "de": de} for en, de in term_pairs],
            }
            out_terms.write(json.dumps(terms_obj, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
