"""Shared utilities for term dictionary building and application."""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"


def repo_rel_path(path: Path | str, *, base: Path = REPO_ROOT) -> str:
    """Return a repo-root-relative path string for portable JSON reports."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(base.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"

DEV_V2_DIR = REPO_ROOT / "shared" / "data" / "dev_v2"
DEV_V1_ORIGINAL_DIR = REPO_ROOT / "shared" / "data" / "dev_v1" / "dev_v1_original"
# Only consumed by experiments/term_expansion/dictionary/scripts/{build_term_dictionary,
# apply_dictionary_to_dev_v1}.py, so it lives under that experiment rather than shared/data/.
TERM_DICTIONARY_DIR = (
    REPO_ROOT / "experiments" / "term_expansion" / "dictionary" / "data" / "dev_v2_dictionary"
)
DEV_V1_DICTIONARY_DIR = REPO_ROOT / "shared" / "data" / "dev_v1" / "dev_v1_dictionary"

SNIPPET_MAX_LEN = 120


@dataclass(frozen=True)
class LangPair:
    prefix: str
    tgt_code: str
    tgt_name: str
    input_name: str
    output_name: str


LANG_PAIRS: dict[str, LangPair] = {
    "ende": LangPair("ende", "de", "German", "ende_dev_v2.jsonl", "ende_term_dictionary.jsonl"),
    "enes": LangPair("enes", "es", "Spanish", "enes_dev_v2.jsonl", "enes_term_dictionary.jsonl"),
    "enru": LangPair("enru", "ru", "Russian", "enru_dev_v2.jsonl", "enru_term_dictionary.jsonl"),
}


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


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def refuse_if_exists(path: Path, *, force: bool) -> None:
    if path.exists() and not force:
        raise SystemExit(
            f"Output already exists: {path}\n"
            "Refusing to overwrite. Pass --force to regenerate."
        )


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


def lemma_key(text: str, lang_code: str) -> str:
    """Normalized lemma for cross-form matching (case, inflection)."""
    return _lemma_key_cached(str(text).strip(), lang_code)


@lru_cache(maxsize=50000)
def _lemma_key_cached(text: str, lang_code: str) -> str:
    if not text:
        return ""
    return normalize_key(lemmatize_phrase(text, lang_code))


def collect_target_forms(entry: dict[str, Any]) -> list[str]:
    forms: list[str] = []
    seen: set[str] = set()
    for form in list(entry.get("forms_tgt") or []) + [entry.get("tgt_surface", "")]:
        if not isinstance(form, str) or not form:
            continue
        norm = normalize_key(form)
        if norm in seen:
            continue
        seen.add(norm)
        forms.append(form)
    return forms


def collect_en_forms(entry: dict[str, Any]) -> list[str]:
    forms: list[str] = []
    seen: set[str] = set()
    for form in list(entry.get("forms_en") or []) + [entry.get("en_surface", "")]:
        if not isinstance(form, str) or not form:
            continue
        norm = normalize_key(form)
        if norm in seen:
            continue
        seen.add(norm)
        forms.append(form)
    lemma = entry.get("lemma_en")
    if isinstance(lemma, str) and lemma:
        norm = normalize_key(lemma)
        if norm not in seen:
            seen.add(norm)
            forms.append(lemma)
    return forms


def find_best_substring_match(forms: list[str], haystack: str) -> str | None:
    """Return the longest form that appears as a substring in haystack."""
    for form in sorted(forms, key=len, reverse=True):
        span = locate_substring(form, haystack)
        if span is not None:
            return span
    return None


def entry_matches_reference(
    entry: dict[str, Any],
    reference: str,
) -> bool:
    return find_best_substring_match(collect_target_forms(entry), reference) is not None


def make_snippet(text: str, max_len: int = SNIPPET_MAX_LEN) -> str:
    text = " ".join(str(text).split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def word_count(text: str) -> int:
    return len(str(text).split())


def is_valid_short_key(key: str) -> bool:
    key = key.strip()
    if len(key) > 2:
        return True
    return key.isupper() and key.isalpha()


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


def remove_nested_overlaps(terms: dict[str, str]) -> dict[str, str]:
    """Drop shorter keys that are substrings of longer keys (case-insensitive)."""
    if not terms:
        return {}
    ranked = sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)
    kept: dict[str, str] = {}
    kept_norm: list[str] = []
    for src, tgt in ranked:
        src_norm = normalize_key(src)
        if any(src_norm in longer for longer in kept_norm):
            continue
        kept[src] = tgt
        kept_norm.append(src_norm)
    return kept


def filter_term_pair(
    en_surface: str,
    tgt_surface: str,
    *,
    en: str,
    tgt: str,
    max_words: int,
    max_en_chars: int | None = None,
) -> tuple[str, str] | None:
    src_span = locate_substring(en_surface, en)
    tgt_span = locate_substring(tgt_surface, tgt)
    if src_span is None or tgt_span is None:
        return None
    if word_count(src_span) > max_words:
        return None
    if max_en_chars is not None and len(src_span) > max_en_chars:
        return None
    if not is_valid_short_key(src_span):
        return None
    return src_span, tgt_span


def make_batches(
    items: list[tuple[int, dict[str, Any]]],
    batch_size: int,
) -> list[list[tuple[int, dict[str, Any]]]]:
    return [items[i : i + batch_size] for i in range(0, len(items), batch_size)]


@lru_cache(maxsize=4)
def _spacy_model(lang_code: str) -> Any:
    import spacy

    model_map = {
        "en": "en_core_web_sm",
        "de": "de_core_news_sm",
        "es": "es_core_news_sm",
    }
    model_name = model_map.get(lang_code)
    if model_name is None:
        raise ValueError(f"No spaCy model for language code {lang_code!r}")
    return spacy.load(model_name)


@lru_cache(maxsize=1)
def _pymorphy_analyzer() -> Any:
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def lemmatize_phrase(text: str, lang_code: str) -> str:
    text = str(text).strip()
    if not text:
        return ""

    if lang_code == "ru":
        morph = _pymorphy_analyzer()
        tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
        lemmas: list[str] = []
        for token in tokens:
            if re.match(r"\w", token, flags=re.UNICODE):
                parsed = morph.parse(token)
                lemmas.append(parsed[0].normal_form if parsed else token.lower())
            else:
                lemmas.append(token)
        return " ".join(lemmas)

    nlp = _spacy_model(lang_code)
    doc = nlp(text)
    parts: list[str] = []
    for token in doc:
        if token.is_space:
            continue
        if token.is_punct and not token.is_alpha:
            parts.append(token.text)
        else:
            parts.append(token.lemma_.lower() if lang_code == "en" else token.lemma_)
    return " ".join(parts)


def collect_forms(attestations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group attestations by (lang_pair, lemma_en, lemma_tgt) and attach observed forms."""
    groups: dict[tuple[str, str, str], dict[str, Any]] = {}
    for att in attestations:
        key = (att["lang_pair"], att["lemma_en"], att["lemma_tgt"])
        if key not in groups:
            groups[key] = {
                "forms_en": set(),
                "forms_tgt": set(),
            }
        groups[key]["forms_en"].add(att["en_surface"])
        groups[key]["forms_tgt"].add(att["tgt_surface"])

    enriched: list[dict[str, Any]] = []
    for att in attestations:
        key = (att["lang_pair"], att["lemma_en"], att["lemma_tgt"])
        out = dict(att)
        out["forms_en"] = sorted(groups[key]["forms_en"])
        out["forms_tgt"] = sorted(groups[key]["forms_tgt"])
        enriched.append(out)
    return enriched
