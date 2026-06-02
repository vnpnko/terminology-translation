#!/usr/bin/env python3
"""
Remove weak/generic entries from `proper_terms` in JSONL records.

Example:
    python scripts/data_preparation/clean_poor_proper_terms.py \
      --input data/sap_dev/ende_dev_v1.jsonl \
      --output data/sap_dev/ende_dev_v1.cleaned.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

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
SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
ENV_FILE = SCRIPTS_DIR / ".env"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"

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


def load_dotenv(path: Path = ENV_FILE) -> None:
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


def clean_record(
    record: dict,
    drop_terms: frozenset[str],
    context_hints: frozenset[str],
    strict_drop: bool,
    *,
    use_openai: bool,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    explanation: str,
    examples: list[str],
    provider: str,
) -> tuple[dict, int]:
    sentence = str(record.get("en", ""))
    proper_terms = record.get("proper_terms")
    if not isinstance(proper_terms, dict):
        return record, 0

    cleaned = {}
    removed = 0
    openai_remove_set: set[str] = set()
    if use_openai and proper_terms:
        openai_remove_set = choose_terms_to_remove_openai(
            sentence=sentence,
            proper_terms=proper_terms,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            explanation=explanation,
            examples=examples,
            provider=provider,
        )

    for src_term, tgt_term in proper_terms.items():
        if use_openai:
            if str(src_term) in openai_remove_set:
                removed += 1
                continue
            cleaned[src_term] = tgt_term
            continue

        if should_drop_term(
            str(src_term),
            sentence,
            drop_terms,
            context_hints,
            strict_drop,
        ):
            removed += 1
            continue
        cleaned[src_term] = tgt_term

    record["proper_terms"] = cleaned
    return record, removed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Remove poor terminology from JSONL proper_terms."
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL file with an `en` field and `proper_terms` object.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--drop-term",
        action="append",
        default=[],
        help=(
            "Extra term to drop (repeatable). "
            "Defaults already include provided/create/image/use variants."
        ),
    )
    parser.add_argument(
        "--context-hint",
        action="append",
        default=[],
        help="Extra context hint token(s) used to keep terms in domain context.",
    )
    parser.add_argument(
        "--strict-drop",
        action="store_true",
        help="Always remove dropped terms, even when context looks topic-specific.",
    )
    parser.add_argument(
        "--mode",
        choices=("openai", "rule-based"),
        default="openai",
        help="Filtering mode (default: openai).",
    )
    parser.add_argument(
        "--use-openai",
        action="store_true",
        help="Deprecated alias; OpenAI is already the default mode.",
    )
    parser.add_argument(
        "--openai-model",
        default=DEFAULT_OPENAI_MODEL,
        help=f"OpenAI model for --use-openai (default: {DEFAULT_OPENAI_MODEL})",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "openai", "openrouter"),
        default="auto",
        help="LLM provider for openai mode (default: auto).",
    )
    parser.add_argument(
        "--openai-api-key",
        default="",
        help="OpenAI API key (or set OPENAI_API_KEY in scripts/.env).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for API calls.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries for API errors/invalid JSON.",
    )
    parser.add_argument(
        "--explanation",
        default=(
            "Remove terms that are generic verbs/nouns/adjectives and not stable domain terminology. "
            "Keep terms that are clearly technical, product-, workflow-, or module-specific in this sentence."
        ),
        help="Instruction text passed to the model for every line.",
    )
    parser.add_argument(
        "--example",
        action="append",
        default=[],
        help=(
            "Extra example instruction (repeatable), e.g. "
            "'remove provided/use/create unless clearly domain-specific'."
        ),
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if args.max_retries < 1:
        raise ValueError("--max-retries must be >= 1")

    drop_terms = set(DEFAULT_DROP_TERMS)
    drop_terms.update(normalize_term(t) for t in args.drop_term if t.strip())
    drop_terms_frozen = frozenset(drop_terms)
    context_hints = set(DEFAULT_CONTEXT_HINTS)
    context_hints.update(normalize_term(t) for t in args.context_hint if t.strip())
    context_hints_frozen = frozenset(context_hints)
    use_openai = args.mode == "openai" or args.use_openai
    openai_api_key = (
        args.openai_api_key
        or os.environ.get("OPENAI_API_KEY", "")
        or os.environ.get("OPENROUTER_API_KEY", "")
    )
    if use_openai and not openai_api_key:
        raise SystemExit(
            f"Missing API key. Set OPENAI_API_KEY or OPENROUTER_API_KEY in {ENV_FILE}, "
            "or pass --openai-api-key."
        )
    provider = args.provider
    if provider == "auto":
        provider = "openrouter" if openai_api_key.startswith("sk-or-v1-") else "openai"

    total_lines = 0
    total_removed = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.input.open("r", encoding="utf-8") as fin, args.output.open(
        "w", encoding="utf-8"
    ) as fout:
        for line_no, line in enumerate(fin, start=1):
            raw = line.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_no} of {args.input}"
                ) from exc

            cleaned_record, removed = clean_record(
                record,
                drop_terms_frozen,
                context_hints_frozen,
                args.strict_drop,
                use_openai=use_openai,
                api_key=openai_api_key,
                model=args.openai_model,
                timeout=args.timeout,
                max_retries=args.max_retries,
                explanation=args.explanation,
                examples=args.example,
                provider=provider,
            )
            fout.write(json.dumps(cleaned_record, ensure_ascii=False) + "\n")
            total_lines += 1
            total_removed += removed
            if use_openai and total_lines % 25 == 0:
                print(f"Processed {total_lines} lines...")

    print(
        f"Processed {total_lines} lines from {args.input}\n"
        f"Removed {total_removed} poor proper_terms\n"
        f"Wrote cleaned data to {args.output}\n"
        f"Mode: {'openai' if use_openai else 'rule-based'}"
    )


if __name__ == "__main__":
    main()
