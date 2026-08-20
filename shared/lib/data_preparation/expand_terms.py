"""Expand proper_terms in JSONL dev files using OpenRouter GPT-4o-mini.

Writes ``<input_stem>_expand.jsonl`` next to the input file — refuses to run
if that output already exists. Reads ``OPENROUTER_API_KEY`` from ``.env`` at
the repo root. The source sentence, target sentence, and random_terms are
preserved; only new validated EN->target term pairs are appended to
proper_terms.

Usage::

    python expand_terms.py -i ../../data/dev_v1/dev_v1_original/ende_dev_v1.jsonl
    python expand_terms.py -i ../../data/dev_v1/dev_v1_original/enes_dev_v1.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]

ENV_FILE = REPO_ROOT / ".env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

VALID_INPUT_RE = re.compile(r"^(ende|enes|enru)_dev_v\d+$")
REJECTED_STEM_MARKERS = ("_expand", "_cleaned", ".expand", ".cleaned")


@dataclass(frozen=True)
class LangPair:
    prefix: str
    tgt_code: str
    tgt_name: str
    direction: str


LANG_PAIRS: dict[str, LangPair] = {
    "ende": LangPair("ende", "de", "German", "EN→DE"),
    "enes": LangPair("enes", "es", "Spanish", "EN→ES"),
    "enru": LangPair("enru", "ru", "Russian", "EN→RU"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8-sig") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            record = json.loads(raw)
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object per line")
            records.append(record)
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE lines from the repo-root .env without overriding env vars."""
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


def resolve_lang_pair(input_path: Path) -> LangPair:
    stem = input_path.stem
    for marker in REJECTED_STEM_MARKERS:
        if marker in stem:
            raise SystemExit(
                f"Refusing to process files that already have a pipeline suffix "
                f"({marker!r}): {input_path.name}"
            )
    if not VALID_INPUT_RE.match(stem):
        raise SystemExit(
            "Input filename must match <pair>_dev_v<N>.jsonl "
            f"(e.g. ende_dev_v1.jsonl), got: {input_path.name}"
        )
    pair_prefix = stem.split("_dev_", 1)[0]
    lang_pair = LANG_PAIRS.get(pair_prefix)
    if lang_pair is None:
        raise SystemExit(
            f"Unsupported language pair prefix {pair_prefix!r} in {input_path.name}. "
            f"Expected one of: {', '.join(sorted(LANG_PAIRS))}."
        )
    return lang_pair


def output_path_for(input_path: Path) -> Path:
    stem = input_path.stem
    return input_path.parent / f"{stem}_expand{input_path.suffix}"


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


def build_batch_prompt(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    lang_pair: LangPair,
) -> str:
    rows = []
    for idx, record in batch:
        rows.append(
            {
                "id": idx,
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
- Terms from another row. Rows are independent; never borrow terminology across
  row IDs.

Return JSON only in this exact shape:
{{"rows": [
  {{"id": 0, "proper_terms": {{"<exact English surface>": "<exact {lang_pair.tgt_name} surface>"}}}},
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
        {"role": "user", "content": build_batch_prompt(batch, lang_pair=lang_pair)},
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
    *,
    tgt_code: str,
) -> dict[str, str]:
    en = str(record.get("en", ""))
    tgt = str(record.get(tgt_code, ""))
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
        tgt_span = locate_substring(tgt_raw, tgt)
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
    lang_pair: LangPair,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> list[tuple[int, dict[str, Any], int]]:
    candidates_by_idx = extract_batch_candidates(
        batch,
        lang_pair=lang_pair,
        api_key=api_key,
        model=model,
        temperature=temperature,
        timeout=timeout,
        max_retries=max_retries,
    )

    results: list[tuple[int, dict[str, Any], int]] = []
    for idx, record in batch:
        additions = validated_new_terms(
            record,
            candidates_by_idx.get(idx, {}),
            tgt_code=lang_pair.tgt_code,
        )
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
    lang_pair: LangPair,
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
        if "en" not in record or lang_pair.tgt_code not in record:
            raise ValueError(
                f"{input_path}:{idx + 1}: expected 'en' and "
                f"'{lang_pair.tgt_code}' fields"
            )
        indexed_rows.append((idx, record))

    batches = make_batches(indexed_rows, batch_size)
    print(
        f"Processing {len(indexed_rows)} row(s) ({lang_pair.direction}) as "
        f"{len(batches)} batch(es) with {workers} worker(s), "
        f"batch_size={batch_size}"
    )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                expand_batch,
                batch,
                lang_pair=lang_pair,
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
        description=(
            "Increase proper_terms coverage using OpenRouter GPT-4o-mini. "
            "Writes <input_stem>_expand.jsonl next to the input file."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help="Input JSONL path (e.g. data/dev_v1/dev_v1_original/ende_dev_v1.jsonl)",
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

    lang_pair = resolve_lang_pair(input_path)
    output_path = output_path_for(input_path)
    if output_path.exists():
        raise SystemExit(f"Output file already exists: {output_path}")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise SystemExit(
            f"Missing API key. Set OPENROUTER_API_KEY in {ENV_FILE} "
            "or pass --api-key."
        )

    expand_file(
        input_path,
        output_path,
        lang_pair=lang_pair,
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
