"""Rule-based and LLM-judge term-dropping logic for clean_poor_proper_terms.py."""

from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request

DEFAULT_DROP_TERMS = frozenset(
    {
        "provided",
        "provide",
        "provides",
        "providing",
        "create",
        "creates",
        "created",
        "creating",
        "image",
        "images",
        "use",
        "uses",
        "used",
        "using",
    }
)
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Generic context hints that suggest topic-specific/technical usage.
DEFAULT_CONTEXT_HINTS = frozenset(
    {
        "create",
        "created",
        "creating",
        "execute",
        "executing",
        "executed",
        "run",
        "running",
        "schedule",
        "scheduled",
        "query",
        "queries",
        "database",
        "batch",
        "system",
        "process",
        "record",
        "table",
        "service",
        "api",
        "sap",
        "workflow",
        "task",
        "job",
        "jobs",
    }
)


def normalize_term(term: str) -> str:
    return term.strip().casefold()


def singularize_simple(term: str) -> str:
    if term.endswith("s") and len(term) > 3:
        return term[:-1]
    return term


def has_topic_specific_context(
    term: str, sentence: str, context_hints: frozenset[str]
) -> bool:
    sentence_lc = sentence.casefold()
    term_lc = normalize_term(term)
    term_variants = {term_lc, singularize_simple(term_lc), f"{term_lc}s"}

    # If the term is not present as a token in the source sentence, we cannot infer context.
    if not any(re.search(rf"\b{re.escape(v)}\b", sentence_lc) for v in term_variants):
        return False

    tokens = re.findall(r"[a-zA-Z0-9_/-]+", sentence_lc)
    if not tokens:
        return False

    hint_set = {normalize_term(h) for h in context_hints}
    term_token_set = {v for v in term_variants if v}
    term_positions = [i for i, tok in enumerate(tokens) if tok in term_token_set]
    if not term_positions:
        return False

    # Look for technical hints in a local window around each term occurrence.
    for pos in term_positions:
        left = max(0, pos - 4)
        right = min(len(tokens), pos + 5)
        window = tokens[left:right]
        if any(tok in hint_set for tok in window):
            return True

    # Fallback: any technical hint in the sentence indicates likely domain context.
    return any(tok in hint_set for tok in tokens)


def should_drop_term(
    term: str,
    sentence: str,
    drop_terms: frozenset[str],
    context_hints: frozenset[str],
    strict_drop: bool,
) -> bool:
    folded = normalize_term(term)
    folded_singular = singularize_simple(folded)

    if folded not in drop_terms and folded_singular not in drop_terms:
        return False

    if strict_drop:
        return True

    if has_topic_specific_context(term, sentence, context_hints):
        return False

    return True


def openai_chat_completion(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout: float,
    provider: str,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": 0.0,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    if provider == "openrouter":
        url = OPENROUTER_URL
    else:
        url = OPENAI_URL
    req = urllib.request.Request(
        url,
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
        raise RuntimeError(f"Unexpected OpenAI response: {payload}") from exc


def choose_terms_to_remove_openai(
    *,
    sentence: str,
    proper_terms: dict[str, str],
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    explanation: str,
    examples: list[str],
    provider: str,
) -> set[str]:
    term_keys = list(proper_terms.keys())
    if not term_keys:
        return set()

    examples_block = ""
    if examples:
        examples_block = "Examples:\n" + "\n".join(f"- {x}" for x in examples) + "\n\n"

    prompt = f"""Decide which `proper_terms` should be REMOVED as poor/generic terminology.

Keep a term if it is domain/topic-specific in this sentence context, even if it can be generic in other contexts.
Do not invent terms; choose only from the candidate list.
Guidance:
{explanation}

{examples_block}English sentence:
{sentence}

Candidate proper_terms keys:
{json.dumps(term_keys, ensure_ascii=False)}

Return JSON exactly in this format:
{{"remove": ["term1", "term2"]}}
"""
    messages = [
        {"role": "system", "content": "You are a strict terminology quality filter."},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = openai_chat_completion(
                api_key=api_key,
                model=model,
                messages=messages,
                timeout=timeout,
                provider=provider,
            )
            parsed = json.loads(raw.strip())
            remove = parsed.get("remove", [])
            if not isinstance(remove, list):
                raise ValueError("Field 'remove' must be a list.")
            candidate_set = {str(k) for k in term_keys}
            return {str(t) for t in remove if str(t) in candidate_set}
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            RuntimeError,
            ValueError,
        ) as exc:
            last_error = exc
            if attempt == max_retries:
                if isinstance(exc, urllib.error.HTTPError) and exc.code == 401:
                    provider_help = (
                        "OPENAI_API_KEY for OpenAI (starts with sk-), or "
                        "OPENROUTER_API_KEY for OpenRouter (starts with sk-or-v1-)."
                    )
                    raise RuntimeError(
                        f"Unauthorized (401) from {provider}. Check API key/provider. "
                        f"Expected key type: {provider_help}"
                    ) from exc
                raise
            time.sleep(min(2**attempt, 20))
    raise last_error if last_error else RuntimeError("Unknown OpenAI error.")
