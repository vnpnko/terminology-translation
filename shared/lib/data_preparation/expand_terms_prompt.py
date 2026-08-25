"""OpenRouter prompt-building, response-parsing, and validation logic for expand_terms.py."""

from __future__ import annotations

import json
import re
import time
import urllib.request
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
    lang_pair: Any,
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
    lang_pair: Any,
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
