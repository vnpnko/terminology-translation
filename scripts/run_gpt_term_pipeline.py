#!/usr/bin/env python3
"""
3-step GPT term pipeline on dev_v1 (500 lines × 3 language pairs).

Step 1: Extract EN domain terms (English only, no reference translation).
Step 2: Propose target-language translations for those terms (no reference).
Step 3: Translate the sentence using the proposed term pairs.

Usage::

    python scripts/run_gpt_term_pipeline.py --limit 10
    python scripts/run_gpt_term_pipeline.py --all
    python scripts/run_gpt_term_pipeline.py --all --resume
    python scripts/run_gpt_term_pipeline.py --lang ende --steps extract,propose
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

_SCRIPT_DIR = Path(__file__).resolve().parent
_PREP_DIR = _SCRIPT_DIR / "data_preparation"
if str(_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_PREP_DIR))

from term_utils import (  # noqa: E402
    LANG_PAIRS,
    LangPair,
    locate_substring,
    make_batches,
    normalize_key,
    openrouter_chat,
    parse_model_json,
    repo_rel_path,
    remove_nested_overlaps,
)

_BASELINE_PATH = _SCRIPT_DIR / "run_openai_baseline.py"
_spec = importlib.util.spec_from_file_location("run_openai_baseline", _BASELINE_PATH)
_baseline = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_baseline)

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LANG_GROUPS = ["ende", "enru", "enes"]
MODE = "gpt_proposed_term"
DEFAULT_BATCH_SIZE = 10
MIN_TERMS = 1
MAX_TERMS = 2

DEFAULT_INPUT_DIR = REPO_ROOT / "data" / "dev_v1" / "original"
DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "dev_v1" / "gpt_proposed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dev_v1" / "original" / "gpt_pipeline"


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip().lstrip("\ufeff")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def load_jsonl(path: Path, max_samples: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if max_samples is not None and len(records) >= max_samples:
                break
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def cache_stem(lang: str) -> str:
    return f"{lang}_dev_v1_gpt_terms"


def input_stem(lang: str) -> str:
    return f"{lang}_dev_v1"


def gpt_terms_for_sample(sample: dict[str, Any]) -> dict[str, str]:
    terms = sample.get("gpt_proposed_terms") or {}
    if not isinstance(terms, dict):
        return {}
    return {str(k): str(v) for k, v in terms.items() if k and v}


def validate_en_terms(terms: list[str] | dict[str, Any], en: str) -> list[str]:
    if isinstance(terms, dict):
        candidates = list(terms.keys())
    else:
        candidates = list(terms)
    validated: list[str] = []
    seen: set[str] = set()
    for raw in candidates:
        if not isinstance(raw, str):
            continue
        span = locate_substring(raw, en)
        if span is None:
            continue
        norm = normalize_key(span)
        if norm in seen:
            continue
        seen.add(norm)
        validated.append(span)
    return validated[:MAX_TERMS]


def build_extract_prompt(batch: list[tuple[int, dict[str, Any]]]) -> str:
    rows = [{"id": line_id, "en": record.get("en", "")} for line_id, record in batch]
    return f"""You annotate SAP / enterprise IT documentation with domain terminology.

For each row, pick between {MIN_TERMS} and {MAX_TERMS} **proper terms** from the English sentence only:
- IT / business-software vocabulary (products, modules, data objects, workflows, UI labels, technical concepts).
- Prefer the most domain-specific terms, not generic words.
- Each term MUST be copied exactly as it appears in the English sentence (surface form, including capitalization).
- Terms may be one or two words (e.g. "consumption model", "data flow").
- If there is no IT / enterprise terminology, return an empty list.

Return JSON only in this exact shape:
{{"rows": [
  {{"id": 0, "terms": ["consumption model", "space"]}},
  {{"id": 1, "terms": []}}
]}}

Rows:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


def build_propose_prompt(
    batch: list[tuple[int, dict[str, Any], list[str]]],
    *,
    lang_pair: LangPair,
) -> str:
    rows = []
    for line_id, record, terms in batch:
        rows.append(
            {
                "id": line_id,
                "en": record.get("en", ""),
                "terms": terms,
                "target_language": lang_pair.tgt_name,
            }
        )
    return f"""You propose {lang_pair.tgt_name} translations for English domain terms.

For each row:
- Given the English sentence and a list of English terms, propose the best {lang_pair.tgt_name}
  translation for each term in an SAP / enterprise software context.
- Do NOT invent terms that are not in the provided terms list.
- Keys MUST be the exact English term strings from the terms list.
- Values MUST be non-empty {lang_pair.tgt_name} strings.
- If a term should stay in English in {lang_pair.tgt_name} (e.g. product names), repeat the English form.

Return JSON only in this exact shape:
{{"rows": [
  {{"id": 0, "proper_terms": {{"consumption model": "...", "space": "..."}}}},
  {{"id": 1, "proper_terms": {{}}}}
]}}

Rows:
{json.dumps(rows, ensure_ascii=False, indent=2)}
"""


def parse_extract_batch(parsed: dict[str, Any]) -> dict[int, list[str]]:
    rows = parsed.get("rows")
    if not isinstance(rows, list):
        raise ValueError("Model response must contain a rows list")
    out: dict[int, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        terms = row.get("terms") or []
        if not isinstance(row_id, int) or not isinstance(terms, list):
            continue
        out[row_id] = [str(t) for t in terms if isinstance(t, str)]
    return out


def parse_propose_batch(parsed: dict[str, Any]) -> dict[int, dict[str, str]]:
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
            if isinstance(src, str) and isinstance(tgt, str) and tgt.strip()
        }
    return out


def call_llm_json_batch(
    messages: list[dict[str, str]],
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    for attempt in range(max_retries):
        try:
            raw = openrouter_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            return parse_model_json(raw)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))
    raise RuntimeError("LLM batch call failed")


def run_extract_batches(
    records: list[dict[str, Any]],
    *,
    pending_ids: list[int],
    api_key: str,
    model: str,
    batch_size: int,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> None:
    items = [(idx, records[idx]) for idx in pending_ids]
    for batch in tqdm(
        make_batches(items, batch_size),
        desc="extract",
        total=max(1, (len(items) + batch_size - 1) // batch_size),
    ):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a meticulous terminology annotator. "
                    "Return valid JSON only. Treat every row ID independently."
                ),
            },
            {"role": "user", "content": build_extract_prompt(batch)},
        ]
        parsed = call_llm_json_batch(
            messages,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        by_id = parse_extract_batch(parsed)
        for line_id, record in batch:
            en = str(record.get("en", ""))
            terms = validate_en_terms(by_id.get(line_id, []), en)
            records[line_id]["gpt_extracted_terms"] = terms


def run_propose_batches(
    records: list[dict[str, Any]],
    *,
    pending_ids: list[int],
    lang_pair: LangPair,
    api_key: str,
    model: str,
    batch_size: int,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> None:
    items: list[tuple[int, dict[str, Any], list[str]]] = []
    for idx in pending_ids:
        record = records[idx]
        terms = record.get("gpt_extracted_terms") or []
        if not isinstance(terms, list):
            terms = []
        items.append((idx, record, terms))

    for batch in tqdm(
        make_batches(items, batch_size),
        desc="propose",
        total=max(1, (len(items) + batch_size - 1) // batch_size),
    ):
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a meticulous terminology translator. "
                    "Return valid JSON only. Treat every row ID independently."
                ),
            },
            {"role": "user", "content": build_propose_prompt(batch, lang_pair=lang_pair)},
        ]
        parsed = call_llm_json_batch(
            messages,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        by_id = parse_propose_batch(parsed)
        for line_id, record, extracted in batch:
            en = str(record.get("en", ""))
            proposed = by_id.get(line_id, {})
            filtered: dict[str, str] = {}
            allowed = set(extracted)
            for src, tgt in proposed.items():
                span = locate_substring(src, en)
                if span is None or span not in allowed:
                    continue
                if not str(tgt).strip():
                    continue
                filtered[span] = str(tgt).strip()
            records[line_id]["gpt_proposed_terms"] = remove_nested_overlaps(filtered)

    for idx in pending_ids:
        if "gpt_proposed_terms" not in records[idx]:
            records[idx]["gpt_proposed_terms"] = {}


def terminology_accuracy_gpt(
    preds: list[str], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    term_ratios: dict[str, float] = {}
    total_terms = 0
    for pred, sample in zip(preds, samples):
        source_text = sample.get("en", "")
        for src, tgt in gpt_terms_for_sample(sample).items():
            total_terms += 1
            src_count = max(_baseline._count_term_occurrences(source_text, src), 1)
            tgt_count = _baseline._count_term_occurrences(pred, tgt)
            term_ratios[src] = min(tgt_count / src_count, 1.0)
    avg_ratio = sum(term_ratios.values()) / len(term_ratios) * 100 if term_ratios else None
    return {"total_terms": total_terms, "avg_ratio_pct": avg_ratio, "per_term_ratios": term_ratios}


def terminology_consistency_gpt(
    preds: list[str], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    term_to_candidates: dict[str, list[str]] = defaultdict(list)
    for pred, sample in zip(preds, samples):
        for src, tgt in gpt_terms_for_sample(sample).items():
            candidate = tgt if str(tgt).lower() in str(pred).lower() else "<MISSING>"
            term_to_candidates[src].append(candidate)

    per_term = {}
    macro_scores = []
    weighted_scores = []
    for src, candidates in term_to_candidates.items():
        pseudo_ref = Counter(candidates).most_common(1)[0][0]
        matches = sum(1 for c in candidates if c == pseudo_ref)
        consistency = matches / len(candidates)
        per_term[src] = {
            "occ": len(candidates),
            "pseudo_ref": pseudo_ref,
            "matches": matches,
            "consistency": consistency,
        }
        macro_scores.append(consistency)
        weighted_scores.extend([consistency] * len(candidates))

    return {
        "per_term": per_term,
        "macro_avg_consistency": sum(macro_scores) / len(macro_scores) if macro_scores else None,
        "weighted_avg_consistency": (
            sum(weighted_scores) / len(weighted_scores) if weighted_scores else None
        ),
    }


def compute_oracle_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare GPT extraction/proposal against oracle proper_terms."""

    def en_key(term: str) -> str:
        return normalize_key(term)

    extraction_hits = 0
    extraction_total = 0
    proposal_hits = 0
    proposal_total = 0

    for sample in samples:
        oracle = sample.get("proper_terms") or {}
        if not isinstance(oracle, dict):
            continue
        extracted = sample.get("gpt_extracted_terms") or []
        proposed = gpt_terms_for_sample(sample)

        oracle_keys = {en_key(src): (src, tgt) for src, tgt in oracle.items() if src and tgt}
        extracted_keys = {en_key(term) for term in extracted if term}

        for key in oracle_keys:
            extraction_total += 1
            if key in extracted_keys:
                extraction_hits += 1

        for src, oracle_tgt in oracle.items():
            if not src or not oracle_tgt:
                continue
            src_key = en_key(src)
            matched_key = None
            for prop_src in proposed:
                if en_key(prop_src) == src_key:
                    matched_key = prop_src
                    break
            if matched_key is None:
                continue
            proposal_total += 1
            prop_tgt = proposed[matched_key]
            if normalize_key(prop_tgt) == normalize_key(oracle_tgt):
                proposal_hits += 1

    return {
        "extraction_overlap_pct": (
            extraction_hits / extraction_total * 100 if extraction_total else None
        ),
        "extraction_hits": extraction_hits,
        "extraction_total": extraction_total,
        "proposal_match_pct": proposal_hits / proposal_total * 100 if proposal_total else None,
        "proposal_hits": proposal_hits,
        "proposal_total": proposal_total,
    }


def run_translate(
    client: OpenAI,
    model: str,
    max_retries: int,
    retry_delay: float,
    lang: str,
    records: list[dict[str, Any]],
    *,
    pending_ids: list[int] | None = None,
) -> None:
    config = _baseline.LANG_CONFIG[lang]
    output_tag = config["output_tag"]
    ids = pending_ids if pending_ids is not None else list(range(len(records)))

    for idx in tqdm(ids, desc=f"{lang}/translate"):
        record = records[idx]
        terms = gpt_terms_for_sample(record)
        pred = _baseline.translate_sample(
            client,
            model,
            max_retries,
            retry_delay,
            record.get("en", ""),
            terms or None,
            config["target_lang"],
            output_tag,
            lang,
            "proper_term",
        )
        record[f"prediction_{MODE}"] = pred
        record[f"prediction_{MODE}_clean"] = _baseline.strip_output_tags(pred, output_tag)


def pending_for_step(records: list[dict[str, Any]], step: str, resume: bool) -> list[int]:
    if not resume:
        return list(range(len(records)))

    if step == "extract":
        return [
            i
            for i, r in enumerate(records)
            if not isinstance(r.get("gpt_extracted_terms"), list)
        ]
    if step == "propose":
        return [
            i
            for i, r in enumerate(records)
            if isinstance(r.get("gpt_extracted_terms"), list)
            and not isinstance(r.get("gpt_proposed_terms"), dict)
        ]
    if step == "translate":
        return [
            i
            for i, r in enumerate(records)
            if isinstance(r.get("gpt_proposed_terms"), dict)
            and f"prediction_{MODE}_clean" not in r
        ]
    return list(range(len(records)))


def merge_cached_records(
    source: list[dict[str, Any]], cached: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for src, cache in zip(source, cached):
        row = dict(src)
        for key in (
            "gpt_extracted_terms",
            "gpt_proposed_terms",
            f"prediction_{MODE}",
            f"prediction_{MODE}_clean",
        ):
            if key in cache:
                row[key] = cache[key]
        merged.append(row)
    return merged


def evaluate_language(
    lang: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    stem: str,
) -> dict[str, Any]:
    config = _baseline.LANG_CONFIG[lang]
    ref_field = config["ref_field"]
    output_tag = config["output_tag"]

    pred_path = output_dir / f"{stem}_{MODE}_predictions.jsonl"
    save_jsonl(pred_path, records)

    clean_preds = [
        r.get(f"prediction_{MODE}_clean", "")
        for r in records
        if f"prediction_{MODE}_clean" in r
    ]
    refs = [r.get(ref_field, "") for r in records[: len(clean_preds)]]
    samples = records[: len(clean_preds)]

    metrics: dict[str, Any] = {}
    if samples and clean_preds:
        metrics.update(_baseline.compute_bleu_chrf(clean_preds, refs))
        term_acc = terminology_accuracy_gpt(clean_preds, samples)
        term_cons = terminology_consistency_gpt(clean_preds, samples)
        metrics["terminology_accuracy"] = term_acc
        metrics["terminology_consistency"] = term_cons
        metrics["oracle_diagnostics"] = compute_oracle_diagnostics(samples)
        diag = metrics["oracle_diagnostics"]
        print(
            f"[{lang}/{MODE}] BLEU={_baseline.fmt_metric(metrics['bleu'])} "
            f"chrF={_baseline.fmt_metric(metrics['chrf'])} "
            f"term_acc={_baseline.fmt_metric(term_acc['avg_ratio_pct'])}% "
            f"extract_overlap={_baseline.fmt_metric(diag['extraction_overlap_pct'])}% "
            f"proposal_match={_baseline.fmt_metric(diag['proposal_match_pct'])}%"
        )

    return {"predictions_file": repo_rel_path(pred_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run GPT extract → propose → translate pipeline.")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--lang", choices=LANG_GROUPS, default=None, help="Single language pair")
    parser.add_argument("--all", action="store_true", help="Run all language pairs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--steps",
        default="extract,propose,translate",
        help="Comma-separated steps: extract, propose, translate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.all and args.lang is None:
        raise SystemExit("Specify --all or --lang ende|enru|enes")

    load_dotenv(REPO_ROOT / ".env")
    api_key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY in .env")

    model = args.model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    steps = {s.strip() for s in args.steps.split(",") if s.strip()}
    langs = LANG_GROUPS if args.all else [args.lang]
    assert langs is not None

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "pipeline": "gpt_extract_propose_translate",
        "data_dir": repo_rel_path(args.input_dir),
        "cache_dir": repo_rel_path(args.cache_dir),
        "provider": "openrouter",
        "base_url": OPENROUTER_BASE_URL,
        "model": model,
        "max_samples": args.limit,
        "batch_size": args.batch_size,
        "steps": sorted(steps),
        "languages": {},
    }

    for lang in langs:
        lang_pair = LANG_PAIRS[lang]
        input_path = args.input_dir / f"{input_stem(lang)}.jsonl"
        cache_path = args.cache_dir / f"{cache_stem(lang)}.jsonl"
        if not input_path.is_file():
            raise SystemExit(f"Missing dataset: {input_path}")

        source_records = load_jsonl(input_path, args.limit)
        if args.resume and cache_path.is_file():
            cached = load_jsonl(cache_path, args.limit)
            if len(cached) != len(source_records):
                raise SystemExit(
                    f"Cache length mismatch for {lang}: {len(cached)} vs {len(source_records)}"
                )
            records = merge_cached_records(source_records, cached)
        else:
            records = [dict(r) for r in source_records]

        print(f"\n=== {lang}: {len(records)} samples → {lang_pair.tgt_name} ===")

        if "extract" in steps:
            pending = pending_for_step(records, "extract", args.resume)
            if pending:
                run_extract_batches(
                    records,
                    pending_ids=pending,
                    api_key=api_key,
                    model=model,
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
            save_jsonl(cache_path, records)

        if "propose" in steps:
            pending = pending_for_step(records, "propose", args.resume)
            if pending:
                run_propose_batches(
                    records,
                    pending_ids=pending,
                    lang_pair=lang_pair,
                    api_key=api_key,
                    model=model,
                    batch_size=args.batch_size,
                    temperature=args.temperature,
                    timeout=args.timeout,
                    max_retries=args.max_retries,
                )
            save_jsonl(cache_path, records)

        lang_output_dir = args.output_dir / lang
        lang_output_dir.mkdir(parents=True, exist_ok=True)
        stem = input_stem(lang)

        if "translate" in steps:
            pending = pending_for_step(records, "translate", args.resume)
            if pending:
                run_translate(
                    client,
                    model,
                    args.max_retries,
                    args.retry_delay,
                    lang,
                    records,
                    pending_ids=pending,
                )
            save_jsonl(cache_path, records)

        mode_result = evaluate_language(lang, records, lang_output_dir, stem)
        summary["languages"][lang] = {
            "input_file": repo_rel_path(input_path),
            "cache_file": repo_rel_path(cache_path),
            "sample_count": len(records),
            "ref_field": lang_pair.tgt_code,
            "target_lang": lang_pair.tgt_name,
            "modes": {MODE: mode_result},
        }

    metrics_path = args.output_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. {metrics_path}")


if __name__ == "__main__":
    main()
