"""Collect ``proper_terms`` pairs from dev_v1 and dev_v2 JSONL files.

Writes one JSONL file per language to ``data/dev_v2/dev_v2_for_training/``
(default path; override with ``--output``) — never modifies the v1/v2
inputs. Reads v1 from ``data/dev_v1/dev_v1_original/`` and v2 from
``data/dev_v2/dev_v2_original/``. For each language group, merges duplicate
English keys (case-insensitive) across both sources, picks the most
frequent (EN surface, target surface) pair, and writes one JSONL line per
unique term:

  {"en": "<english>", "de": "<german>"}   # or "es" / "ru"

Supported target languages (``-t`` / ``--target-lang``):

  de  —  ``ende_dev_v1.jsonl`` + ``ende_dev_v2.jsonl``
  es  —  ``enes_dev_v1.jsonl`` + ``enes_dev_v2.jsonl`` (v2)
  ru  —  ``enru_dev_v1.jsonl`` + ``enru_dev_v2.jsonl``

Usage::

    python collect_term_pairs_from_jsonl.py -t de
    python collect_term_pairs_from_jsonl.py -t es -o ../../data/dev_v2/dev_v2_for_training/enes_pairs.jsonl
    python collect_term_pairs_from_jsonl.py --all
    python collect_term_pairs_from_jsonl.py -t ru --include-count
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent


@dataclass(frozen=True)
class LangConfig:
    code: str
    pair: str
    field: str
    v1_name: str
    v2_name: str


LANGS: dict[str, LangConfig] = {
    "de": LangConfig(
        code="de",
        pair="ende",
        field="de",
        v1_name="ende_dev_v1.jsonl",
        v2_name="ende_dev_v2.jsonl",
    ),
    "es": LangConfig(
        code="es",
        pair="enes",
        field="es",
        v1_name="enes_dev_v1.jsonl",
        v2_name="enes_dev_v2.jsonl",
    ),
    "ru": LangConfig(
        code="ru",
        pair="enru",
        field="ru",
        v1_name="enru_dev_v1.jsonl",
        v2_name="enru_dev_v2.jsonl",
    ),
}

V1_DIR = REPO_ROOT / "data" / "dev_v1" / "dev_v1_original"
V2_DIR = REPO_ROOT / "data" / "dev_v2" / "dev_v2_original"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "dev_v2" / "dev_v2_for_training"


def default_paths(lang: LangConfig) -> tuple[Path, Path]:
    return V1_DIR / lang.v1_name, V2_DIR / lang.v2_name


def default_output(lang: LangConfig) -> Path:
    return DEFAULT_OUT_DIR / f"{lang.pair}_proper_term_pairs.jsonl"


def collect_pairs_from_jsonl(path: Path) -> Counter[tuple[str, str]]:
    pairs: Counter[tuple[str, str]] = Counter()
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON") from exc

            proper = record.get("proper_terms")
            if not proper:
                continue
            if not isinstance(proper, dict):
                print(
                    f"Warning: {path.name}:{line_no} proper_terms is not an object",
                    file=sys.stderr,
                )
                continue

            for en_key, tgt_val in proper.items():
                if not isinstance(en_key, str) or not isinstance(tgt_val, str):
                    continue
                en = en_key.strip()
                tgt = tgt_val.strip()
                if en and tgt:
                    pairs[(en, tgt)] += 1

    return pairs


def merge_pairs(pair_counts: Counter[tuple[str, str]]) -> list[tuple[str, str, int]]:
    """
    One entry per case-insensitive English key.

    Keeps the (en, target) pair with the highest occurrence count among all
    spelling variants sharing the same English key.
    """
    by_en_fold: dict[str, Counter[tuple[str, str]]] = defaultdict(Counter)
    for (en, tgt), count in pair_counts.items():
        by_en_fold[en.casefold()][(en, tgt)] += count

    merged: list[tuple[str, str, int]] = []
    for variant_counts in by_en_fold.values():
        (en, tgt), total = variant_counts.most_common(1)[0]
        merged.append((en, tgt, total))

    merged.sort(key=lambda row: row[0].casefold())
    return merged


def build_records(
    merged: list[tuple[str, str, int]],
    lang: LangConfig,
    *,
    include_count: bool,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for en, tgt, count in merged:
        row: dict[str, Any] = {"en": en, lang.field: tgt}
        if include_count:
            row["count"] = count
        records.append(row)
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def collect_language(
    lang: LangConfig,
    *,
    v1_path: Path | None,
    v2_path: Path | None,
    include_count: bool,
) -> list[dict[str, Any]]:
    v1 = v1_path or default_paths(lang)[0]
    v2 = v2_path or default_paths(lang)[1]

    total: Counter[tuple[str, str]] = Counter()
    for label, path in (("v1", v1), ("v2", v2)):
        if not path.is_file():
            raise FileNotFoundError(f"Missing {label} file for {lang.pair}: {path}")
        file_pairs = collect_pairs_from_jsonl(path)
        total.update(file_pairs)
        print(
            f"  [{label}] {path.name}: {len(file_pairs)} unique pairs, "
            f"{sum(file_pairs.values())} occurrences"
        )

    merged = merge_pairs(total)
    print(f"  merged: {len(merged)} unique English keys ({lang.pair})")
    return build_records(merged, lang, include_count=include_count)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect proper_terms EN→target pairs from SAP v1+v2 into JSONL."
    )
    parser.add_argument(
        "-t",
        "--target-lang",
        choices=sorted(LANGS),
        default=None,
        help="Target language: de, es, or ru (required unless --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Write JSONL for de, es, and ru",
    )
    parser.add_argument(
        "--v1",
        type=Path,
        default=None,
        help="Override v1 JSONL (only with a single -t)",
    )
    parser.add_argument(
        "--v2",
        type=Path,
        default=None,
        help="Override v2 JSONL (only with a single -t)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (default: data/dev_v2/dev_v2_for_training/<pair>_proper_term_pairs.jsonl)",
    )
    parser.add_argument(
        "--include-count",
        action="store_true",
        help='Add integer "count" field (occurrences in v1+v2)',
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.all and args.target_lang:
        raise SystemExit("Use either -t or --all, not both.")
    if not args.all and not args.target_lang:
        raise SystemExit("Specify -t de|es|ru or --all.")

    targets = list(LANGS.values()) if args.all else [LANGS[args.target_lang]]

    for lang in targets:
        print(f"\n{lang.pair} ({lang.code}):")
        records = collect_language(
            lang,
            v1_path=args.v1.resolve() if args.v1 and len(targets) == 1 else None,
            v2_path=args.v2.resolve() if args.v2 and len(targets) == 1 else None,
            include_count=args.include_count,
        )
        out_path = (
            args.output.resolve()
            if args.output and len(targets) == 1
            else default_output(lang)
        )
        write_jsonl(out_path, records)
        print(f"Wrote {len(records)} lines -> {out_path}")


if __name__ == "__main__":
    main()
