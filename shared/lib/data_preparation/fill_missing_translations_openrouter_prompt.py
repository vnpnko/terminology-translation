"""OpenRouter prompt-building and translation logic for fill_missing_translations_openrouter.py."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from typing import Any

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
TERM_FIELDS = frozenset({"proper_terms", "random_terms"})


def build_prompt(
    en_text: str,
    proper_terms: dict[str, str],
    random_terms: dict[str, str],
    target_lang: str,
) -> str:
    proper_keys = list(proper_terms.keys())
    random_keys = list(random_terms.keys())
    return f"""You are a professional translator for technical / enterprise software documentation.

Translate the English source into {target_lang}.

Return a single JSON object with exactly these keys:
- "sentence": the full translated sentence (preserve trailing newlines from the source if present)
- "proper_terms": object mapping each English proper-term key to its {target_lang} translation as used in the translated sentence
- "random_terms": object mapping each English random-term key to its {target_lang} translation as used in the translated sentence

Rules:
1. Use terminology consistent with the sentence translation.
2. Keep proper_terms and random_terms keys identical to the English keys listed below (do not add or remove keys).
3. For multi-word keys, translate the phrase as it appears in context.
4. Output JSON only, no markdown fences.

English sentence:
{en_text}

proper_terms keys (translate each value): {json.dumps(proper_keys, ensure_ascii=False)}

random_terms keys (translate each value): {json.dumps(random_keys, ensure_ascii=False)}
"""


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


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


def translate_record(
    record: dict[str, Any],
    *,
    tgt_code: str,
    target_lang: str,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> dict[str, Any]:
    proper_terms = dict(record.get("proper_terms") or {})
    random_terms = dict(record.get("random_terms") or {})
    en_text = record["en"]

    prompt = build_prompt(en_text, proper_terms, random_terms, target_lang)
    messages = [
        {
            "role": "system",
            "content": "You respond with valid JSON only.",
        },
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = openrouter_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            parsed = parse_model_json(raw)
            break
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 30))
    else:
        raise last_error  # pragma: no cover

    sentence = parsed.get("sentence", "")
    if not isinstance(sentence, str):
        raise ValueError(f"Invalid sentence in model response: {sentence!r}")

    out = dict(record)
    out[tgt_code] = sentence

    for field in TERM_FIELDS:
        source_keys = dict(record.get(field) or {})
        model_terms = parsed.get(field) or {}
        if not isinstance(model_terms, dict):
            raise ValueError(f"Invalid {field} in model response: {model_terms!r}")
        filled: dict[str, str] = {}
        for key in source_keys:
            value = model_terms.get(key, "")
            if not isinstance(value, str):
                value = str(value) if value is not None else ""
            filled[key] = value
        out[field] = filled

    return out
