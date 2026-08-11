#!/usr/bin/env python3
"""
Remove weak/generic entries from `proper_terms` in JSONL records.

Writes ``<pair>_dev_v<N>_cleaned.jsonl`` next to the input file.

Usage::

    python clean_poor_proper_terms.py -i ../../data/interim/dev_v1_expand/ende_dev_v1_expand.jsonl
    python clean_poor_proper_terms.py -i ../../data/raw/dev_v1_original/ende_dev_v1.jsonl
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
REPO_ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = REPO_ROOT / ".env"
DEFAULT_MODEL = "gpt-4o-mini"

VALID_INPUT_RE = re.compile(r"^(ende|enes|enru)_dev_v\d+(_expand)?$")
REJECTED_STEM_MARKERS = ("_cleaned", ".cleaned")


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
            "Input filename must match <pair>_dev_v<N>.jsonl or "
            f"<pair>_dev_v<N>_expand.jsonl, got: {input_path.name}"
        )
    pair_prefix = stem.split("_dev_", 1)[0]
    lang_pair = LANG_PAIRS.get(pair_prefix)
    if lang_pair is None:
        raise SystemExit(
            f"Unsupported language pair prefix {pair_prefix!r} in {input_path.name}. "
            f"Expected one of: {', '.join(sorted(LANG_PAIRS))}."
        )
    return lang_pair


def cleaned_stem_from(input_stem: str) -> str:
    if input_stem.endswith("_expand"):
        return f"{input_stem[: -len('_expand')]}_cleaned"
    return f"{input_stem}_cleaned"


def output_path_for(input_path: Path) -> Path:
    return input_path.parent / f"{cleaned_stem_from(input_path.stem)}{input_path.suffix}"


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


def make_batches(
    items: list[tuple[int, dict[str, Any]]],
    batch_size: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


def clean_batch(
    batch: list[tuple[int, dict[str, Any]]],
    *,
    drop_terms: frozenset[str],
    context_hints: frozenset[str],
    strict_drop: bool,
    use_openai: bool,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    explanation: str,
    examples: list[str],
    provider: str,
) -> list[tuple[int, dict[str, Any], int]]:
    results: list[tuple[int, dict[str, Any], int]] = []
    for idx, record in batch:
        cleaned_record, removed = clean_record(
            record,
            drop_terms,
            context_hints,
            strict_drop,
            use_openai=use_openai,
            api_key=api_key,
            model=model,
            timeout=timeout,
            max_retries=max_retries,
            explanation=explanation,
            examples=examples,
            provider=provider,
        )
        results.append((idx, cleaned_record, removed))
    return results


def clean_file(
    input_path: Path,
    output_path: Path,
    *,
    lang_pair: LangPair,
    drop_terms: frozenset[str],
    context_hints: frozenset[str],
    strict_drop: bool,
    use_openai: bool,
    api_key: str,
    model: str,
    timeout: float,
    max_retries: int,
    explanation: str,
    examples: list[str],
    provider: str,
    limit: int | None,
    workers: int,
    batch_size: int,
) -> None:
    records = load_jsonl(input_path)
    cleaned_records = list(records)
    total_removed = 0
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
                clean_batch,
                batch,
                drop_terms=drop_terms,
                context_hints=context_hints,
                strict_drop=strict_drop,
                use_openai=use_openai,
                api_key=api_key,
                model=model,
                timeout=timeout,
                max_retries=max_retries,
                explanation=explanation,
                examples=examples,
                provider=provider,
            )
            for batch in batches
        ]
        for done, future in enumerate(as_completed(futures), start=1):
            batch_results = future.result()
            batch_removed = 0
            for idx, new_record, removed in batch_results:
                cleaned_records[idx] = new_record
                total_removed += removed
                batch_removed += removed
            print(
                f"[batch {done}/{len(batches)}] "
                f"rows={len(batch_results)}, removed={batch_removed}"
            )

    save_jsonl(output_path, cleaned_records)
    print(f"Wrote {len(cleaned_records)} rows to {output_path}")
    print(f"Total poor proper terms removed: {total_removed}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Remove poor terminology from JSONL proper_terms. "
            "Writes <pair>_dev_v<N>_cleaned.jsonl next to the input file."
        )
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        required=True,
        help=(
            "Input JSONL path (e.g. data/interim/dev_v1_expand/ende_dev_v1_expand.jsonl "
            "or data/raw/dev_v1_original/ende_dev_v1.jsonl)"
        ),
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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--openai-model",
        default=None,
        help="Deprecated alias for --model.",
    )
    parser.add_argument(
        "--provider",
        choices=("auto", "openai", "openrouter"),
        default="auto",
        help="LLM provider for openai mode (default: auto).",
    )
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--openai-api-key",
        default=None,
        help="Deprecated alias for --api-key.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds for API calls.",
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=5)
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

    input_path = args.input.resolve()
    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")
    if args.workers < 1:
        raise SystemExit("--workers must be >= 1")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be >= 1")
    if args.max_retries < 1:
        raise SystemExit("--max-retries must be >= 1")

    lang_pair = resolve_lang_pair(input_path)
    output_path = output_path_for(input_path)
    if output_path.exists():
        raise SystemExit(f"Output file already exists: {output_path}")

    drop_terms = set(DEFAULT_DROP_TERMS)
    drop_terms.update(normalize_term(t) for t in args.drop_term if t.strip())
    drop_terms_frozen = frozenset(drop_terms)
    context_hints = set(DEFAULT_CONTEXT_HINTS)
    context_hints.update(normalize_term(t) for t in args.context_hint if t.strip())
    context_hints_frozen = frozenset(context_hints)
    use_openai = args.mode == "openai" or args.use_openai
    model = args.openai_model or args.model
    api_key = (
        args.api_key
        or args.openai_api_key
        or os.environ.get("OPENROUTER_API_KEY", "")
        or os.environ.get("OPENAI_API_KEY", "")
    )
    if use_openai and not api_key:
        raise SystemExit(
            f"Missing API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY in {ENV_FILE} "
            "or pass --api-key."
        )
    provider = args.provider
    if provider == "auto":
        provider = "openrouter" if api_key.startswith("sk-or-v1-") else "openai"

    clean_file(
        input_path,
        output_path,
        lang_pair=lang_pair,
        drop_terms=drop_terms_frozen,
        context_hints=context_hints_frozen,
        strict_drop=args.strict_drop,
        use_openai=use_openai,
        api_key=api_key,
        model=model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        explanation=args.explanation,
        examples=args.example,
        provider=provider,
        limit=args.limit,
        workers=args.workers,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
