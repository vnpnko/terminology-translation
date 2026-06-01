from __future__ import annotations

import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

TAG_BY_LANGUAGE = {
    "german": "deu",
    "de": "deu",
    "deu": "deu",
    "spanish": "es",
    "es": "es",
    "russian": "ru",
    "ru": "ru",
}


def load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: str | os.PathLike[str], records: list[dict[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def terminology_for_mode(sample: dict[str, Any], mode: str) -> dict[str, str] | None:
    if mode == "no_term":
        return None
    if mode == "proper_term":
        terminology = (sample.get("proper_terms") or {}).copy()
        return terminology or None
    if mode == "random_term":
        terminology = (sample.get("random_terms") or {}).copy()
        for key in (sample.get("proper_terms") or {}).keys():
            terminology.pop(key, None)
        return terminology or None
    return None


def output_tag_for_language(target_lang: str) -> str:
    return TAG_BY_LANGUAGE.get(str(target_lang).strip().lower(), "deu")


def strip_output_tags(text: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(
        r"</?(deu|de|ger|german|es|spa|spanish|ru|rus|russian)>",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()


def build_translation_prompt(
    source_text: str,
    target_lang: str,
    terminology: dict[str, str] | None = None,
) -> str:
    output_tag = output_tag_for_language(target_lang)
    term_block = ""

    if terminology:
        term_block = "Terminology:\n"
        for src, tgt in terminology.items():
            term_block += f"{src} -> {tgt}\n"
        term_block += "\n"

    return f"""
Translate the English text to {target_lang}.

Rules:
1. Output only in this format: <{output_tag}> ... </{output_tag}>
2. Use the terminology mappings exactly as provided.
3. Do not explain anything.
4. Translate only from English to {target_lang}.

{term_block}
Input:
<en> {source_text} </en>
"""


def compute_bleu_chrf(hyps: list[str], refs: list[str]) -> dict[str, float]:
    import sacrebleu

    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {
        "bleu": bleu.score,
        "chrf": chrf.score,
    }


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _count_term_occurrences(term: str, text: str) -> int:
    text_norm = _normalize_text(text)
    term_norm = _normalize_text(term)
    pattern = r"\b" + re.escape(term_norm) + r"\b"
    return len(re.findall(pattern, text_norm))


def terminology_accuracy_advanced(
    samples: list[dict[str, Any]],
    predictions: list[str],
    mode: str = "proper_term",
) -> dict[str, Any]:
    term_ratios: dict[str, float] = {}
    total_terms = 0

    for sample, pred in zip(samples, predictions):
        if mode == "proper_term":
            terms = (sample.get("proper_terms") or {}).copy()
        elif mode == "random_term":
            terms = (sample.get("random_terms") or {}).copy()
            for key in (sample.get("proper_terms") or {}).keys():
                terms.pop(key, None)
        else:
            terms = {}

        source_text = sample.get("en", "")

        for src, tgt in terms.items():
            total_terms += 1

            src_count = _count_term_occurrences(src, source_text)
            if src_count == 0:
                src_count = 1

            tgt_count = _count_term_occurrences(tgt, pred)
            ratio = min(tgt_count / src_count, 1.0)

            term_ratios[src] = ratio

    avg_accuracy = (
        sum(term_ratios.values()) / len(term_ratios) * 100
        if term_ratios
        else None
    )

    return {
        "total_terms": total_terms,
        "avg_ratio_pct": avg_accuracy,
        "per_term_ratios": term_ratios,
    }


def terminology_consistency_advanced(
    samples: list[dict[str, Any]],
    predictions: list[str],
    mode: str = "proper_term",
) -> dict[str, Any]:
    term_to_candidates: dict[str, list[str]] = defaultdict(list)

    for sample, pred in zip(samples, predictions):
        if mode == "proper_term":
            terms = (sample.get("proper_terms") or {}).copy()
        elif mode == "random_term":
            terms = (sample.get("random_terms") or {}).copy()
            for key in (sample.get("proper_terms") or {}).keys():
                terms.pop(key, None)
        else:
            terms = {}

        for src, tgt in terms.items():
            if str(tgt).lower() in str(pred).lower():
                term_to_candidates[src].append(tgt)
            else:
                term_to_candidates[src].append("<MISSING>")

    pseudo_references: dict[str, str] = {}

    for src, candidates in term_to_candidates.items():
        counter = Counter(candidates)
        pseudo_references[src] = counter.most_common(1)[0][0]

    per_term_consistency: dict[str, dict[str, Any]] = {}
    macro_scores: list[float] = []
    weighted_scores: list[float] = []

    for src, candidates in term_to_candidates.items():
        pseudo_ref = pseudo_references[src]
        matches = sum(1 for candidate in candidates if candidate == pseudo_ref)
        consistency = matches / len(candidates) if candidates else 0.0

        per_term_consistency[src] = {
            "occ": len(candidates),
            "pseudo_ref": pseudo_ref,
            "matches": matches,
            "consistency": consistency,
        }

        macro_scores.append(consistency)
        weighted_scores.extend([consistency] * len(candidates))

    return {
        "per_term": per_term_consistency,
        "macro_avg_consistency": (
            sum(macro_scores) / len(macro_scores)
            if macro_scores
            else None
        ),
        "weighted_avg_consistency": (
            sum(weighted_scores) / len(weighted_scores)
            if weighted_scores
            else None
        ),
    }
