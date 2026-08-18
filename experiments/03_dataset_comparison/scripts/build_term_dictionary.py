"""Build a provenance-aware term dictionary from dev_v2 JSONL files.

Writes ``<pair>_term_dictionary.jsonl`` and a combined ``build_report.json``
to ``experiments/03_dataset_comparison/data/dev_v2_dictionary/`` (default; use
``--force`` to overwrite). Reads dev_v2 input from ``data/dev_v2/``. Uses
OpenRouter for LLM-assisted extraction (``OPENROUTER_API_KEY`` in
``.env``); pass ``--skip-llm`` to seed from human ``proper_terms`` only.

  1. Lines 1-1000: seed from human ``proper_terms``, then expand via LLM.
  2. Lines 1001-2000: LLM extraction only (ignores existing ``proper_terms``).

Usage::

    python build_term_dictionary.py --all --limit 50
    python build_term_dictionary.py -t de
    python build_term_dictionary.py --all --force
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_preparation.term_utils import (
    DEFAULT_MODEL,
    DEV_V2_DIR,
    ENV_FILE,
    LANG_PAIRS,
    LangPair,
    TERM_DICTIONARY_DIR,
    collect_forms,
    filter_term_pair,
    lemmatize_phrase,
    load_dotenv,
    load_jsonl,
    locate_substring,
    make_batches,
    make_snippet,
    normalize_key,
    openrouter_chat,
    parse_model_json,
    refuse_if_exists,
    remove_nested_overlaps,
    save_json,
    save_jsonl,
)

TRUSTED_MAX_LINE = 1000
TRUSTED_MAX_WORDS = 3
UNTRUSTED_MAX_WORDS = 2
UNTRUSTED_MAX_EN_CHARS = 30


def build_expand_prompt(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    lang_pair: LangPair,
) -> str:
    rows = []
    for line_id, record in batch:
        rows.append(
            {
                "id": line_id,
                "en": record.get("en", ""),
                lang_pair.tgt_code: record.get(lang_pair.tgt_code, ""),
                "proper_terms": record.get("proper_terms") or {},
                "random_terms": record.get("random_terms") or {},
            }
        )
    return f"""You expand terminology annotations for SAP / enterprise software documentation.

Task:
For each row, find as many domain-relevant proper terms as possible in that
row's English sentence and align each to the exact {lang_pair.tgt_name} surface
text used in that same row's {lang_pair.tgt_name} translation.

Proper terms include:
- SAP, enterprise software, IT, database, platform, data-model, workflow, UI,
  business-process, accounting, analytics, configuration, security, integration,
  document, table/view/procedure, app/module/service, and technical concept terms.
- Multi-word terms are preferred when they form a specific concept.

Do NOT include:
- Generic function words or ordinary adjectives/verbs unless part of a domain term.
- Terms already present in that row's existing proper_terms or random_terms.
- A term if either the English surface or {lang_pair.tgt_name} surface is not
  copied verbatim from that same row's provided sentences.
- Terms from another row. Rows are independent; never borrow terminology across row IDs.

Return JSON only in this exact shape:
{{"rows": [
  {{"id": 0, "proper_terms": {{"<exact English surface>": "<exact {lang_pair.tgt_name} surface>"}}}},
  {{"id": 1, "proper_terms": {{}}}}
]}}

Rows:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


def build_extract_prompt(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    lang_pair: LangPair,
) -> str:
    rows = []
    for line_id, record in batch:
        rows.append(
            {
                "id": line_id,
                "en": record.get("en", ""),
                lang_pair.tgt_code: record.get(lang_pair.tgt_code, ""),
            }
        )
    return f"""You annotate SAP / enterprise IT documentation sentence pairs with domain terminology.

For each row, pick between 1 and 3 **proper terms**:
- IT / business-software vocabulary (products, modules, data objects, workflows, UI labels, technical concepts).
- Prefer the most domain-specific terms, not generic words (avoid: the, can, use, new, etc.).
- Each English key MUST be copied exactly as it appears in the English sentence (surface form, including capitalization).
- Each {lang_pair.tgt_name} value MUST be copied exactly as it appears in the {lang_pair.tgt_name} sentence.
- Terms may be one or two words (e.g. "consumption model", "data flow"). Do NOT return phrases longer than 2 words.
- Do NOT return long app names or UI titles longer than 30 characters.
- If there is no IT / enterprise terminology (generic UI or help text only), return an empty object.

Return JSON only in this exact shape:
{{"rows": [
  {{"id": 0, "proper_terms": {{"<english surface>": "<{lang_pair.tgt_name.lower()} surface>"}}}},
  {{"id": 1, "proper_terms": {{}}}}
]}}

Rows:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


def parse_batch_terms(parsed: dict[str, Any]) -> dict[int, dict[str, str]]:
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Model response must contain a rows list")

    out: dict[int, dict[str, str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        terms = row.get("proper_terms") or {}
        if not isinstance(row_id, int) or not isinstance(terms, dict):
            continue
        out[row_id] = {
            str(src): str(tgt)
            for src, tgt in terms.items()
            if isinstance(src, str) and isinstance(tgt, str)
        }
    return out


def call_llm_batch(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    prompt_builder: Callable[..., str],
    lang_pair: LangPair,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> dict[int, dict[str, str]]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous terminology annotator. "
                "Return valid JSON only. Treat every row ID independently."
            ),
        },
        {"role": "user", "content": prompt_builder(batch, lang_pair=lang_pair)},
    ]
    for attempt in range(max_retries):
        try:
            raw = openrouter_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            return parse_batch_terms(parse_model_json(raw))
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
    return {}


def terms_to_attestations(
    terms: dict[str, str],
    *,
    record: dict[str, Any],
    line_id: int,
    lang_pair: LangPair,
    source_file: str,
    source: str,
    max_words: int,
    max_en_chars: int | None,
    blocked_keys: set[str] | None = None,
) -> list[dict[str, Any]]:
    en = str(record.get("en", ""))
    tgt = str(record.get(lang_pair.tgt_code, ""))
    blocked = blocked_keys or set()

    filtered: dict[str, str] = {}
    for src_raw, tgt_raw in terms.items():
        pair = filter_term_pair(
            src_raw,
            tgt_raw,
            en=en,
            tgt=tgt,
            max_words=max_words,
            max_en_chars=max_en_chars,
        )
        if pair is None:
            continue
        src_span, tgt_span = pair
        if normalize_key(src_span) in blocked:
            continue
        filtered[src_span] = tgt_span

    filtered = remove_nested_overlaps(filtered)
    attestations: list[dict[str, Any]] = []
    for en_surface, tgt_surface in filtered.items():
        attestations.append(
            {
                "lang_pair": lang_pair.prefix,
                "line_id": line_id,
                "source_file": source_file,
                "en_surface": en_surface,
                "tgt_surface": tgt_surface,
                "lemma_en": lemmatize_phrase(en_surface, "en"),
                "lemma_tgt": lemmatize_phrase(tgt_surface, lang_pair.tgt_code),
                "en_snippet": make_snippet(en),
                "tgt_snippet": make_snippet(tgt),
                "source": source,
            }
        )
    return attestations


def seed_human_terms(
    records: list[dict[str, Any]],
    *,
    lang_pair: LangPair,
    source_file: str,
    line_start: int,
    line_end: int,
) -> list[dict[str, Any]]:
    attestations: list[dict[str, Any]] = []
    for line_id in range(line_start, min(line_end, len(records)) + 1):
        record = records[line_id - 1]
        proper = record.get("proper_terms") or {}
        if not isinstance(proper, dict):
            continue
        attestations.extend(
            terms_to_attestations(
                proper,
                record=record,
                line_id=line_id,
                lang_pair=lang_pair,
                source_file=source_file,
                source="human_proper_terms",
                max_words=TRUSTED_MAX_WORDS,
                max_en_chars=None,
            )
        )
    return attestations


def llm_extract_for_range(
    records: list[dict[str, Any]],
    line_ids: list[int],
    *,
    lang_pair: LangPair,
    source_file: str,
    prompt_builder: Callable[..., str],
    source_label: str,
    max_words: int,
    max_en_chars: int | None,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    workers: int,
    batch_size: int,
    include_blocked_from_record: bool,
) -> list[dict[str, Any]]:
    if not line_ids:
        return []

    indexed: list[tuple[int, dict[str, Any]]] = [
        (line_id, records[line_id - 1]) for line_id in line_ids
    ]
    batches = make_batches(indexed, batch_size)
    print(
        f"  LLM ({source_label}): {len(line_ids)} rows, "
        f"{len(batches)} batches, workers={workers}"
    )

    candidates_by_line: dict[int, dict[str, str]] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                call_llm_batch,
                batch,
                prompt_builder=prompt_builder,
                lang_pair=lang_pair,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            ): batch
            for batch in batches
        }
        for done, future in enumerate(as_completed(futures), start=1):
            batch = futures[future]
            result = future.result()
            candidates_by_line.update(result)
            print(f"    [batch {done}/{len(futures)}] rows={len(batch)}")

    attestations: list[dict[str, Any]] = []
    for line_id in line_ids:
        record = records[line_id - 1]
        blocked: set[str] = set()
        if include_blocked_from_record:
            for field in ("proper_terms", "random_terms"):
                terms = record.get(field) or {}
                if isinstance(terms, dict):
                    blocked.update(normalize_key(k) for k in terms if isinstance(k, str))

        attestations.extend(
            terms_to_attestations(
                candidates_by_line.get(line_id, {}),
                record=record,
                line_id=line_id,
                lang_pair=lang_pair,
                source_file=source_file,
                source="llm_extract",
                max_words=max_words,
                max_en_chars=max_en_chars,
                blocked_keys=blocked,
            )
        )
    return attestations


def dedupe_attestations(attestations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for att in attestations:
        key = (
            att["lang_pair"],
            att["line_id"],
            normalize_key(att["en_surface"]),
            normalize_key(att["tgt_surface"]),
            att["source"],
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(att)
    return out


def validate_attestations(
    attestations: list[dict[str, Any]],
    records: list[dict[str, Any]],
    *,
    lang_pair: LangPair,
) -> tuple[list[dict[str, Any]], int]:
    valid: list[dict[str, Any]] = []
    invalid = 0
    for att in attestations:
        line_id = att["line_id"]
        if line_id < 1 or line_id > len(records):
            invalid += 1
            continue
        record = records[line_id - 1]
        en = str(record.get("en", ""))
        tgt = str(record.get(lang_pair.tgt_code, ""))
        if (
            locate_substring(att["en_surface"], en) is None
            or locate_substring(att["tgt_surface"], tgt) is None
        ):
            invalid += 1
            continue
        valid.append(att)
    return valid, invalid


def build_report(
    attestations: list[dict[str, Any]],
    *,
    lang_pair: LangPair,
    invalid_count: int,
) -> dict[str, Any]:
    by_source: Counter[str] = Counter()
    by_range: Counter[str] = Counter()
    for att in attestations:
        by_source[att["source"]] += 1
        rng = "1-1000" if att["line_id"] <= TRUSTED_MAX_LINE else "1001-2000"
        by_range[rng] += 1

    lemma_groups = len(
        {(a["lemma_en"], a["lemma_tgt"]) for a in attestations}
    )
    return {
        "lang_pair": lang_pair.prefix,
        "total_attestations": len(attestations),
        "invalid_filtered": invalid_count,
        "unique_lemma_pairs": lemma_groups,
        "by_source": dict(by_source),
        "by_line_range": dict(by_range),
    }


def build_dictionary_for_pair(
    lang_pair: LangPair,
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    limit: int | None,
    workers: int,
    batch_size: int,
    force: bool,
    skip_llm: bool,
) -> dict[str, Any]:
    input_path = DEV_V2_DIR / lang_pair.input_name
    output_path = TERM_DICTIONARY_DIR / lang_pair.output_name
    refuse_if_exists(output_path, force=force)

    if not input_path.is_file():
        raise SystemExit(f"Input not found: {input_path}")

    records = load_jsonl(input_path)
    max_line = limit if limit is not None else len(records)
    max_line = min(max_line, len(records))

    trusted_end = min(TRUSTED_MAX_LINE, max_line)
    untrusted_start = TRUSTED_MAX_LINE + 1
    untrusted_end = max_line

    print(f"\n=== {lang_pair.prefix} ({lang_pair.tgt_name}) ===")
    print(f"Processing lines 1-{max_line} from {input_path.name}")

    attestations: list[dict[str, Any]] = []

    if trusted_end >= 1:
        print(f"  Seeding human terms (lines 1-{trusted_end})...")
        attestations.extend(
            seed_human_terms(
                records,
                lang_pair=lang_pair,
                source_file=lang_pair.input_name,
                line_start=1,
                line_end=trusted_end,
            )
        )
        print(f"    {len(attestations)} attestations from human proper_terms")

    if not skip_llm and trusted_end >= 1:
        trusted_ids = list(range(1, trusted_end + 1))
        expand_atts = llm_extract_for_range(
            records,
            trusted_ids,
            lang_pair=lang_pair,
            source_file=lang_pair.input_name,
            prompt_builder=build_expand_prompt,
            source_label="expand 1-1000",
            max_words=TRUSTED_MAX_WORDS,
            max_en_chars=None,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            workers=workers,
            batch_size=batch_size,
            include_blocked_from_record=True,
        )
        print(f"    +{len(expand_atts)} from LLM expansion")
        attestations.extend(expand_atts)

    if not skip_llm and untrusted_end >= untrusted_start:
        untrusted_ids = list(range(untrusted_start, untrusted_end + 1))
        extract_atts = llm_extract_for_range(
            records,
            untrusted_ids,
            lang_pair=lang_pair,
            source_file=lang_pair.input_name,
            prompt_builder=build_extract_prompt,
            source_label="extract 1001+",
            max_words=UNTRUSTED_MAX_WORDS,
            max_en_chars=UNTRUSTED_MAX_EN_CHARS,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
            workers=workers,
            batch_size=batch_size,
            include_blocked_from_record=False,
        )
        print(f"    +{len(extract_atts)} from LLM extraction (1001+)")
        attestations.extend(extract_atts)

    attestations = dedupe_attestations(attestations)
    attestations, invalid_count = validate_attestations(
        attestations, records, lang_pair=lang_pair
    )
    attestations = collect_forms(attestations)
    report = build_report(attestations, lang_pair=lang_pair, invalid_count=invalid_count)

    save_jsonl(output_path, attestations)
    print(f"Wrote {len(attestations)} attestations to {output_path}")
    print(f"  Report: {report}")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build term dictionary from dev_v2 with provenance and lemmas."
    )
    parser.add_argument(
        "-t",
        "--target-lang",
        choices=["de", "es", "ru"],
        help="Target language code (default: process one pair via --all)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all three language pairs",
    )
    parser.add_argument("--model", default=os.environ.get("DICT_EXTRACT_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Max dev_v2 lines to process")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing dictionary output files",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Seed human terms only (no OpenRouter calls)",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.all and args.target_lang is None:
        raise SystemExit("Specify --all or -t de|es|ru")
    if args.workers < 1 or args.batch_size < 1:
        raise SystemExit("--workers and --batch-size must be >= 1")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if not args.skip_llm and not api_key:
        raise SystemExit(
            f"Missing API key. Set OPENROUTER_API_KEY in {ENV_FILE} "
            "or pass --api-key / --skip-llm."
        )

    target_to_prefix = {"de": "ende", "es": "enes", "ru": "enru"}
    if args.all:
        pairs = [LANG_PAIRS["ende"], LANG_PAIRS["enes"], LANG_PAIRS["enru"]]
    else:
        pairs = [LANG_PAIRS[target_to_prefix[args.target_lang]]]

    reports: dict[str, Any] = {}
    for lang_pair in pairs:
        reports[lang_pair.prefix] = build_dictionary_for_pair(
            lang_pair,
            api_key=api_key,
            model=args.model,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            limit=args.limit,
            workers=args.workers,
            batch_size=args.batch_size,
            force=args.force,
            skip_llm=args.skip_llm,
        )

    report_path = TERM_DICTIONARY_DIR / "build_report.json"
    if report_path.exists() and not args.force:
        existing = json.loads(report_path.read_text(encoding="utf-8"))
        if isinstance(existing, dict):
            existing.update(reports)
            reports = existing
    save_json(report_path, reports)
    print(f"\nWrote combined report to {report_path}")


if __name__ == "__main__":
    main()
