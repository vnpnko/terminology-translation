"""Build JSONL outputs from SAP term_postedits flat files.

Reads ``term_postedits.test.<pair>.{src,pe,mt,terms}`` flat files from
``--data-dir`` (default: ``data/sap_term_postedits/<ende|enes|enru>`` — this
raw source is not checked into the repo after the open-sourcing restructure,
so ``--data-dir`` must point at real flat files to run) and writes two JSONL
files back into that same directory: sentences with ``proper_terms``, and a
terms-only file.

Usage::

    python convert_term_postedits_to_jsonl.py -t de --data-dir "../../data/sap_term_postedits/ende"
    python convert_term_postedits_to_jsonl.py -t ru --data-dir "../../data/sap_term_postedits/enru"
    python convert_term_postedits_to_jsonl.py -t es --data-dir path/to/enes --out-jsonl out.jsonl --out-terms-jsonl terms.jsonl
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TargetLang:
    code: str
    name: str
    field: str
    pair: str  # file suffix: ende, enes, enru


TARGET_LANGS: dict[str, TargetLang] = {
    "de": TargetLang("de", "German", "de", "ende"),
    "es": TargetLang("es", "Spanish", "es", "enes"),
    "ru": TargetLang("ru", "Russian", "ru", "enru"),
}

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "sap_term_postedits"


def load_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def parse_terms_line(line: str) -> list[tuple[str, str]]:
    stripped = line.strip()
    if not stripped:
        return []
    try:
        parsed = ast.literal_eval(stripped)
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"Invalid terms line: {stripped}") from exc
    if not isinstance(parsed, list):
        return []
    terms: list[tuple[str, str]] = []
    for item in parsed:
        if not isinstance(item, tuple) or len(item) < 3:
            continue
        en_term, _mt_term, target_term = item[0], item[1], item[2]
        if isinstance(en_term, str) and isinstance(target_term, str):
            terms.append((en_term, target_term))
    return terms


def stem_for(lang: TargetLang) -> str:
    return f"term_postedits.test.{lang.pair}"


def default_data_dir(lang: TargetLang) -> Path:
    return DEFAULT_DATA_ROOT / lang.pair


def default_out_jsonl(data_dir: Path, lang: TargetLang) -> Path:
    return data_dir / f"{stem_for(lang)}.jsonl"


def default_out_terms_jsonl(data_dir: Path, lang: TargetLang) -> Path:
    return data_dir / f"{stem_for(lang)}.terms.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build JSONL outputs with proper terms from term_postedits files."
    )
    parser.add_argument(
        "-t",
        "--target-lang",
        required=True,
        choices=sorted(TARGET_LANGS),
        metavar="LANG",
        help="Target language: de (German), es (Spanish), ru (Russian)",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help=(
            "Directory with term_postedits.test.<pair>.* files "
            "(default: data/sap_term_postedits/<ende|enes|enru>)"
        ),
    )
    parser.add_argument(
        "--out-jsonl",
        type=Path,
        default=None,
        help="Output JSONL with sentences and terms (default: <data-dir>/term_postedits.test.<pair>.jsonl)",
    )
    parser.add_argument(
        "--out-terms-jsonl",
        type=Path,
        default=None,
        help="Output JSONL with term pairs only (default: <data-dir>/term_postedits.test.<pair>.terms.jsonl)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    lang = TARGET_LANGS[args.target_lang]
    data_dir = (args.data_dir or default_data_dir(lang)).resolve()
    out_jsonl = (args.out_jsonl or default_out_jsonl(data_dir, lang)).resolve()
    out_terms_jsonl = (args.out_terms_jsonl or default_out_terms_jsonl(data_dir, lang)).resolve()

    stem = stem_for(lang)
    src_path = data_dir / f"{stem}.src"
    pe_path = data_dir / f"{stem}.pe"
    mt_path = data_dir / f"{stem}.mt"
    terms_path = data_dir / f"{stem}.terms"

    for path in (src_path, pe_path, mt_path, terms_path):
        if not path.is_file():
            raise SystemExit(f"Missing input file: {path}")

    src_lines = load_lines(src_path)
    pe_lines = load_lines(pe_path)
    mt_lines = load_lines(mt_path)
    terms_lines = load_lines(terms_path)

    if not (len(src_lines) == len(pe_lines) == len(mt_lines) == len(terms_lines)):
        raise ValueError(
            "Line counts do not match: "
            f"src={len(src_lines)}, pe={len(pe_lines)}, mt={len(mt_lines)}, terms={len(terms_lines)}"
        )

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_terms_jsonl.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Building {lang.name} JSONL ({lang.pair}) from {data_dir} "
        f"-> {out_jsonl.name}, {out_terms_jsonl.name}"
    )

    with out_jsonl.open("w", encoding="utf-8") as out_sentences, out_terms_jsonl.open(
        "w", encoding="utf-8"
    ) as out_terms:
        for idx, (src, pe, _mt, terms_line) in enumerate(
            zip(src_lines, pe_lines, mt_lines, terms_lines), start=1
        ):
            term_pairs = parse_terms_line(terms_line)
            proper_terms = {en: target for en, target in term_pairs}

            sentence_obj = {
                "en": src,
                lang.field: pe,
                "proper_terms": proper_terms,
                "random_terms": {},
            }
            out_sentences.write(json.dumps(sentence_obj, ensure_ascii=False) + "\n")

            terms_obj = {
                "row": idx,
                "terms": [{"en": en, lang.field: target} for en, target in term_pairs],
            }
            out_terms.write(json.dumps(terms_obj, ensure_ascii=False) + "\n")

    print(f"Wrote {len(src_lines)} records.")


if __name__ == "__main__":
    main()
