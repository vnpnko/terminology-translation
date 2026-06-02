#!/usr/bin/env python3
"""
Expand proper_terms in EN-DE SAP JSONL data using OpenRouter GPT-4o-mini.

The source sentence, target sentence, and random_terms are preserved. Only new
validated EN->DE term pairs are appended to proper_terms.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.core import load_jsonl, save_jsonl

ENV_FILE = REPO_ROOT / "scripts" / ".env"
DEFAULT_INPUT = REPO_ROOT / "data" / "sap_dev" / "ende_dev_v1.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data" / "sap_data_expanded"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE lines from scripts/.env without overriding env vars."""
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


def output_path_for(input_path: Path, output_dir: Path) -> Path:
    return output_dir / f"{input_path.stem}_expand{input_path.suffix}"


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError("Model response is not a JSON object")
    return parsed


def locate_substring(needle: str, haystack: str) -> str | None:
    needle = str(needle).strip()
    haystack = str(haystack)
    if not needle or not haystack:
        return None
    match = re.search(re.escape(needle), haystack, flags=re.IGNORECASE)
    if not match:
        return None
    return haystack[match.start() : match.end()]


def normalize_key(text: str) -> str:
    return " ".join(str(text).casefold().split())


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {payload}") from exc


def build_batch_prompt(batch: list[tuple[int, dict[str, Any]]]) -> str:
    rows = []
    for idx, record in batch:
        rows.append(
            {
                "id": idx,
                "en": record.get("en", ""),
                "de": record.get("de", ""),
                "proper_terms": record.get("proper_terms") or {},
                "random_terms": record.get("random_terms") or {},
            }
        )

    return f"""You expand terminology annotations for SAP / enterprise software documentation.

Task:
For each row, find as many domain-relevant proper terms as possible in that
row's English sentence and align each to the exact German surface text used in
that same row's German translation.

Proper terms include:
- SAP, enterprise software, IT, database, platform, data-model, workflow, UI,
  business-process, accounting, analytics, configuration, security, integration,
  document, table/view/procedure, app/module/service, and technical concept terms.
- Multi-word terms are preferred when they form a specific concept.

Do NOT include:
- Generic function words or ordinary adjectives/verbs unless part of a domain term.
- Terms already present in that row's existing proper_terms or random_terms.
- A term if either the English surface or German surface is not copied verbatim
  from that same row's provided sentences.
- Terms from another row. Rows are independent; never borrow terminology across
  row IDs.

Return JSON only in this exact shape:
{{"rows": [
  {{"id": 0, "proper_terms": {{"<exact English surface>": "<exact German surface>"}}}},
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


def extract_batch_candidates(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> dict[str, str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a meticulous terminology annotator. "
                "Return valid JSON only. Treat every row ID independently."
            ),
        },
        {"role": "user", "content": build_batch_prompt(batch)},
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
            parsed = parse_model_json(raw)
            return parse_batch_terms(parsed)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(min(2 ** (attempt + 1), 30))


def validated_new_terms(
    record: dict[str, Any],
    candidates: dict[str, str],
) -> dict[str, str]:
    en = str(record.get("en", ""))
    de = str(record.get("de", ""))
    proper_terms = record.get("proper_terms") or {}
    random_terms = record.get("random_terms") or {}

    blocked = {
        normalize_key(key)
        for terms in (proper_terms, random_terms)
        if isinstance(terms, dict)
        for key in terms
        if isinstance(key, str)
    }

    additions: dict[str, str] = {}
    for src_raw, tgt_raw in candidates.items():
        src_span = locate_substring(src_raw, en)
        tgt_span = locate_substring(tgt_raw, de)
        if src_span is None or tgt_span is None:
            continue

        src_norm = normalize_key(src_span)
        if not src_norm or src_norm in blocked or src_norm in {
            normalize_key(k) for k in additions
        }:
            continue

        additions[src_span] = tgt_span

    return additions


def expand_batch(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> list[tuple[int, dict[str, Any], int]]:
    candidates_by_idx = extract_batch_candidates(
        batch,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )

    results: list[tuple[int, dict[str, Any], int]] = []
    for idx, record in batch:
        additions = validated_new_terms(record, candidates_by_idx.get(idx, {}))
        out = dict(record)
        proper_terms = dict(record.get("proper_terms") or {})
        proper_terms.update(additions)
        out["proper_terms"] = proper_terms
        results.append((idx, out, len(additions)))
    return results


def make_batches(
    items: list[tuple[int, dict[str, Any]]],
    batch_size: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def expand_file(
    input_path: Path,
    output_path: Path,
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    limit: int | None,
    workers: int,
    batch_size: int,
) -> None:
    records = load_jsonl(input_path)
    expanded = list(records)
    total_added = 0
    rows_to_process = records if limit is None else records[:limit]

    indexed_rows: list[tuple[int, dict[str, Any]]] = []
    for idx, record in enumerate(rows_to_process):
        if "en" not in record or "de" not in record:
            raise ValueError(f"{input_path}:{idx + 1}: expected 'en' and 'de' fields")
        indexed_rows.append((idx, record))

    batches = make_batches(indexed_rows, batch_size)
    print(
        f"Processing {len(indexed_rows)} row(s) as {len(batches)} batch(es) "
        f"with {workers} worker(s), batch_size={batch_size}"
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                expand_batch,
                batch,
                api_key=api_key,
                model=model,
                temperature=temperature,
                timeout=timeout,
                max_retries=max_retries,
            )
            for batch in batches
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            batch_results = future.result()
            batch_added = 0
            for idx, new_record, added in batch_results:
                expanded[idx] = new_record
                total_added += added
                batch_added += added
            print(
                f"[batch {done}/{len(batches)}] "
                f"rows={len(batch_results)}, added={batch_added}"
            )

    save_jsonl(output_path, expanded)
    print(f"Wrote {len(expanded)} rows to {output_path}")
    print(f"Total new proper terms added: {total_added}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Increase EN-DE proper_terms coverage using OpenRouter GPT-4o-mini."
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Input EN-DE JSONL (default: {DEFAULT_INPUT.relative_to(REPO_ROOT)})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output JSONL (default: data/sap_data_expanded/<input_stem>_expand.jsonl)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")

    output_path = (
        args.output.resolve()
        if args.output
        else output_path_for(input_path, DEFAULT_OUTPUT_DIR).resolve()
    )

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set OPENROUTER_API_KEY in {ENV_FILE} "
            "or pass --api-key."
        )

    expand_file(
        input_path,
        output_path,
        api_key=api_key,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        limit=args.limit,
        workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
