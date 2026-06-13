#!/usr/bin/env python3
"""
Run GPT-4o-mini baseline via OpenRouter (same logic as Baseline/Ivan_opneai_baseline.ipynb).

Usage::

    python scripts/run_openai_baseline.py \\
        --data-dir data/dev_v1/dictionary \\
        --data-variant dictionary \\
        --output-dir results/dev_v1/dictionary/gpt

    python scripts/run_openai_baseline.py \\
        --data-dir data/dev_v1/original \\
        --output-dir results/dev_v1/original/gpt \\
        --limit 10
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sacrebleu
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

_PREP_DIR = Path(__file__).resolve().parent / "data_preparation"
if str(_PREP_DIR) not in sys.path:
    sys.path.insert(0, str(_PREP_DIR))
from term_utils import repo_rel_path

REPO_ROOT = Path(__file__).resolve().parents[1]
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

LANG_GROUPS = ["ende", "enru", "enes"]
MODES = ["no_term", "proper_term", "random_term"]

LANG_CONFIG = {
    "ende": {"ref_field": "de", "target_lang": "German", "output_tag": "de"},
    "enru": {"ref_field": "ru", "target_lang": "Russian", "output_tag": "ru"},
    "enes": {"ref_field": "es", "target_lang": "Spanish", "output_tag": "es"},
}

SAMPLE_SENTENCES: dict[str, list[dict[str, object]]] = {
    "ende": [
        {
            "en": "The status of each individual space can be seen from the color code on the upper left corner.",
            "de": "Der Farbcode oben links gibt den Status des betreffenden Space an.",
            "proper_terms": {"space": "Space"},
            "random_terms": {"status": "Status"},
        },
        {
            "en": "This service describes the deployed (run-time) state of SAP HANA database artifacts, for example: tables, views, or procedures, which have been created or adjusted by the SAP Integrated Development Environment (WebIDE) editors as a family of consistent design-time artifacts for all key SAP HANA platform database features.",
            "de": "Dieser Service beschreibt den implementierten Zustand (Laufzeitzustand) von SAP-HANA-Datenbankartefakten, z. B. Tabellen, Views oder Prozeduren, die von den SAP-Integrated-Development-Environment-Editoren (WebIDE-Editoren) als eine Familie konsistenter Entwurfszeit-Artefakte für alle wichtigen SAP-HANA-Plattform-Datenbankfunktionen erstellt oder angepasst wurden.",
            "proper_terms": {"design": "Entwurf", "state": "Zustand"},
            "random_terms": {"service": "Service", "features": "funktionen"},
        },
        {
            "en": "Your contract includes a total number of available capacity units per month which you can allocate as you wish to the compute and storage resources.",
            "de": "Ihr Vertrag enthält eine Gesamtzahl verfügbarer Kapazitätseinheiten pro Monat, die Sie den Rechen- und Speicherressourcen nach Belieben zuweisen können.",
            "proper_terms": {"storage": "Speicher"},
            "random_terms": {"total": "Gesamtzahl", "available": "verfügbarer"},
        },
    ],
    "enru": [
        {
            "en": "To add notes, choose a template and open the Notes pane on the left side of the screen.",
            "ru": "Чтобы добавить примечания, выберите шаблон и откройте область Примечания на левой стороне экрана.",
            "proper_terms": {"pane": "область"},
            "random_terms": {"add": "добавить", "choose": "выберите"},
        },
        {
            "en": "Indicates if a configuration item or configuration step is specific to a localized solution version.",
            "ru": "Указывает, являются ли позиция или шаг конфигурации специфичными для локализованной версии решения.",
            "proper_terms": {"item": "позиция"},
            "random_terms": {"version": "версии", "solution": "решения"},
        },
        {
            "en": "Specifies the number of the contract from which you can select service items.",
            "ru": "Указывает номер контракта, из которого можно выбрать позиции услуг.",
            "proper_terms": {"contract": "контракт"},
            "random_terms": {"number": "номер", "select": "выбрать"},
        },
    ],
    "enes": [
        {
            "en": "Why would you need to access HDI containers?",
            "es": "¿Por qué tendría que acceder a los containers HDI?",
            "proper_terms": {"container": "container"},
            "random_terms": {"access": "acceder"},
        },
        {
            "en": "In such cases you may use the Move Items or Merge feature.",
            "es": "En estos casos, puede utilizar la función Mover elementos o Fusionar .",
            "proper_terms": {"item": "elemento"},
            "random_terms": {"may": "puede"},
        },
        {
            "en": "Decide if you want to use parallel processing for this job:",
            "es": "Decida si desea utilizar el procesamiento paralelo para este job:",
            "proper_terms": {
                "processing": "procesamiento",
                "job": "job",
                "parallel processing": "procesamiento paralelo",
            },
            "random_terms": {"Decide": "Decida", "want": "desea"},
        },
    ],
}


def data_stem(lang: str, data_version: str, data_variant: str | None) -> str:
    stem = f"{lang}_dev_{data_version}"
    if data_variant:
        stem = f"{stem}_{data_variant}"
    return stem


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


def terms_for_mode(sample: dict[str, Any], mode: str) -> dict[str, str]:
    if mode == "proper_term":
        return (sample.get("proper_terms") or {}).copy()
    if mode == "random_term":
        terms = (sample.get("random_terms") or {}).copy()
        for key in (sample.get("proper_terms") or {}):
            terms.pop(key, None)
        return terms
    return {}


def terminology_for_mode(sample: dict[str, Any], mode: str) -> dict[str, str] | None:
    terms = terms_for_mode(sample, mode)
    return terms or None


def strip_output_tags(text: str, output_tag: str) -> str:
    if not isinstance(text, str):
        return text
    return re.sub(rf"</?{re.escape(output_tag)}>", "", text, flags=re.IGNORECASE).strip()


def compute_bleu_chrf(hyps: list[str], refs: list[str]) -> dict[str, float]:
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu": bleu.score, "chrf": chrf.score}


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _count_term_occurrences(text: str, term: str) -> int:
    text_norm = _normalize_text(text)
    term_norm = _normalize_text(term)
    return len(re.findall(r"\b" + re.escape(term_norm) + r"\b", text_norm))


def terminology_accuracy(
    preds: list[str], samples: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    term_ratios: dict[str, float] = {}
    total_terms = 0
    for pred, sample in zip(preds, samples):
        source_text = sample.get("en", "")
        for src, tgt in terms_for_mode(sample, mode).items():
            total_terms += 1
            src_count = max(_count_term_occurrences(source_text, src), 1)
            tgt_count = _count_term_occurrences(pred, tgt)
            term_ratios[src] = min(tgt_count / src_count, 1.0)
    avg_ratio = sum(term_ratios.values()) / len(term_ratios) * 100 if term_ratios else None
    return {"total_terms": total_terms, "avg_ratio_pct": avg_ratio, "per_term_ratios": term_ratios}


def terminology_consistency(
    preds: list[str], samples: list[dict[str, Any]], mode: str
) -> dict[str, Any]:
    term_to_candidates: dict[str, list[str]] = defaultdict(list)
    for pred, sample in zip(preds, samples):
        for src, tgt in terms_for_mode(sample, mode).items():
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


def fmt_metric(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def format_terminology_block(terms: dict[str, str]) -> str:
    if not terms:
        return ""
    return "Terminology:\n" + "\n".join(f"{s} -> {t}" for s, t in terms.items()) + "\n"


def format_sample_examples(lang: str, mode: str) -> str:
    config = LANG_CONFIG[lang]
    ref_field = config["ref_field"]
    output_tag = config["output_tag"]
    blocks = []
    for i, example in enumerate(SAMPLE_SENTENCES[lang], 1):
        term_block = format_terminology_block(terms_for_mode(example, mode))
        ref = example.get(ref_field, "")
        blocks.append(
            f"Example {i}:\n"
            f"{term_block}"
            f"Input:\n<en> {example['en']} </en>\n"
            f"Output:\n<{output_tag}> {ref} </{output_tag}>"
        )
    return "Examples:\n\n" + "\n".join(blocks) + "\n\n"


def translate_sample(
    client: OpenAI,
    model: str,
    max_retries: int,
    retry_delay: float,
    sample_en: str,
    terminology: dict[str, str] | None,
    target_lang: str,
    output_tag: str,
    lang: str,
    mode: str,
) -> str:
    examples_block = format_sample_examples(lang, mode)
    term_block = format_terminology_block(terminology or {})
    if term_block:
        term_block += "\n"

    prompt = f"""You are a translation assistant.

Translate the English text to {target_lang}.

Rules:
1. Output only in this format: <{output_tag}> ... </{output_tag}>
2. Use the terminology mappings exactly as provided.
3. Do not explain anything.
4. Translate only from English to {target_lang}.

{examples_block}{term_block}Input:
<en> {sample_en} </en>
"""

    messages = [
        {"role": "system", "content": "You are a helpful translation assistant."},
        {"role": "user", "content": prompt},
    ]
    last_error: Exception | None = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
    raise RuntimeError(f"OpenRouter API failed after {max_retries} attempts") from last_error


def run_mode(
    client: OpenAI,
    model: str,
    max_retries: int,
    retry_delay: float,
    lang: str,
    mode: str,
    samples: list[dict[str, Any]],
    output_dir: Path,
    config: dict[str, str],
    stem: str,
) -> dict[str, Any]:
    ref_field = config["ref_field"]
    output_tag = config["output_tag"]
    preds: list[str] = []
    records: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc=f"{lang}/{mode}"):
        pred = translate_sample(
            client,
            model,
            max_retries,
            retry_delay,
            sample.get("en", ""),
            terminology_for_mode(sample, mode),
            config["target_lang"],
            output_tag,
            lang,
            mode,
        )
        preds.append(pred)
        record = sample.copy()
        record[f"prediction_{mode}"] = pred
        record[f"prediction_{mode}_clean"] = strip_output_tags(pred, output_tag)
        records.append(record)

    clean_preds = [strip_output_tags(p, output_tag) for p in preds]
    pred_path = output_dir / f"{stem}_{mode}_predictions.jsonl"
    save_jsonl(pred_path, records)

    metrics: dict[str, Any] = {}
    if samples and ref_field in samples[0]:
        refs = [sample.get(ref_field, "") for sample in samples]
        metrics.update(compute_bleu_chrf(clean_preds, refs))
        term_acc = terminology_accuracy(clean_preds, samples, mode)
        term_cons = terminology_consistency(clean_preds, samples, mode)
        metrics["terminology_accuracy"] = term_acc
        metrics["terminology_consistency"] = term_cons
        print(
            f"[{lang}/{mode}] BLEU={fmt_metric(metrics['bleu'])} "
            f"chrF={fmt_metric(metrics['chrf'])} "
            f"term_acc={fmt_metric(term_acc['avg_ratio_pct'])}% "
            f"macro_cons={fmt_metric(term_cons['macro_avg_consistency'])} "
            f"weighted_cons={fmt_metric(term_cons['weighted_avg_consistency'])}"
        )
    return {"predictions_file": repo_rel_path(pred_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenAI/OpenRouter GPT baseline.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=REPO_ROOT / "data" / "dev_v1" / "dictionary",
    )
    parser.add_argument("--data-version", default="v1")
    parser.add_argument("--data-variant", default="dictionary")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "dev_v1" / "dictionary" / "gpt",
    )
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()

    load_dotenv(REPO_ROOT / ".env")
    api_key = (os.environ.get("OPENROUTER_API_KEY") or os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise SystemExit("Missing OPENROUTER_API_KEY in .env")

    model = args.model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-4o-mini")
    client = OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL)

    datasets: dict[str, list[dict[str, Any]]] = {}
    for lang in LANG_GROUPS:
        path = data_dir / f"{data_stem(lang, args.data_version, args.data_variant)}.jsonl"
        if not path.is_file():
            raise SystemExit(f"Missing dataset: {path}")
        datasets[lang] = load_jsonl(path, args.limit)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "data_version": args.data_version,
        "data_variant": args.data_variant,
        "data_dir": repo_rel_path(data_dir),
        "prompt_examples_per_lang": {lang: len(SAMPLE_SENTENCES[lang]) for lang in LANG_GROUPS},
        "provider": "openrouter",
        "base_url": OPENROUTER_BASE_URL,
        "model": model,
        "max_samples": args.limit,
        "languages": {},
    }

    for lang in LANG_GROUPS:
        config = LANG_CONFIG[lang]
        samples = datasets[lang]
        lang_dir = output_dir / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        stem = data_stem(lang, args.data_version, args.data_variant)
        print(f"\n=== {lang}: {len(samples)} samples → {config['target_lang']} ===")
        lang_results = {
            mode: run_mode(
                client,
                model,
                args.max_retries,
                args.retry_delay,
                lang,
                mode,
                samples,
                lang_dir,
                config,
                stem,
            )
            for mode in MODES
        }
        summary["languages"][lang] = {
            "data_file": repo_rel_path(data_dir / f"{stem}.jsonl"),
            "sample_count": len(samples),
            **config,
            "modes": lang_results,
        }

    metrics_path = output_dir / "metrics_summary.json"
    metrics_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nDone. {metrics_path}")


if __name__ == "__main__":
    main()
