"""Compute BLEU/chrF/terminology metrics for the cached GPT-proposed-term pipeline run.

Reads the cached extract -> propose -> translate results from
``experiments/05_gpt_proposed_terms/data/dev_v1_gpt_proposed/{lang}_dev_v1_gpt_terms.jsonl`` (500 lines x
ende/enru/enes, already containing ``gpt_extracted_terms``, ``gpt_proposed_terms``
and ``prediction_gpt_proposed_term(_clean)``). No API calls are made here -- this
only scores predictions that were generated earlier.

Writes:
    results/dev_v1/original/zero_shot/gpt_pipeline/{lang}_dev_v1_gpt_proposed_term_predictions.jsonl
    results/dev_v1/original/zero_shot/gpt_pipeline/metrics_summary.json

Usage::

    python experiments/05_gpt_proposed_terms/scripts/compute_gpt_proposed_metrics.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sacrebleu

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "data_preparation"))

from term_utils import normalize_key, repo_rel_path  # noqa: E402

MODE = "gpt_proposed_term"
LANG_CONFIG = {
    "ende": {"ref_field": "de", "target_lang": "German", "output_tag": "de"},
    "enru": {"ref_field": "ru", "target_lang": "Russian", "output_tag": "ru"},
    "enes": {"ref_field": "es", "target_lang": "Spanish", "output_tag": "es"},
}
LANG_GROUPS = tuple(LANG_CONFIG)

DEFAULT_CACHE_DIR = EXPERIMENT_DIR / "data" / "dev_v1_gpt_proposed"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "results" / "dev_v1" / "original" / "zero_shot" / "gpt_pipeline"


def cache_path(cache_dir: Path, lang: str) -> Path:
    return cache_dir / f"{lang}_dev_v1_gpt_terms.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def gpt_terms_for_sample(sample: dict[str, Any]) -> dict[str, str]:
    terms = sample.get("gpt_proposed_terms") or {}
    return terms if isinstance(terms, dict) else {}


def compute_bleu_chrf(hyps: list[str], refs: list[str]) -> dict[str, float]:
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    return {"bleu": bleu.score, "chrf": chrf.score}


def _normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def _count_term_occurrences(text: str, term: str) -> int:
    import re

    text_norm = _normalize_text(text)
    term_norm = _normalize_text(term)
    return len(re.findall(r"\b" + re.escape(term_norm) + r"\b", text_norm))


def terminology_accuracy_gpt(preds: list[str], samples: list[dict[str, Any]]) -> dict[str, Any]:
    term_ratios: dict[str, float] = {}
    total_terms = 0
    for pred, sample in zip(preds, samples):
        source_text = sample.get("en", "")
        for src, tgt in gpt_terms_for_sample(sample).items():
            total_terms += 1
            src_count = max(_count_term_occurrences(source_text, src), 1)
            tgt_count = _count_term_occurrences(pred, tgt)
            term_ratios[src] = min(tgt_count / src_count, 1.0)
    avg_ratio = sum(term_ratios.values()) / len(term_ratios) * 100 if term_ratios else None
    return {"total_terms": total_terms, "avg_ratio_pct": avg_ratio, "per_term_ratios": term_ratios}


def terminology_consistency_gpt(preds: list[str], samples: list[dict[str, Any]]) -> dict[str, Any]:
    term_to_candidates: dict[str, list[str]] = defaultdict(list)
    for pred, sample in zip(preds, samples):
        for src, tgt in gpt_terms_for_sample(sample).items():
            candidate = tgt if str(tgt).lower() in str(pred).lower() else "<MISSING>"
            term_to_candidates[src].append(candidate)

    per_term: dict[str, Any] = {}
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


def compute_oracle_diagnostics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare GPT's self-extracted/self-proposed terms against the oracle proper_terms."""
    extraction_hits = 0
    extraction_total = 0
    proposal_hits = 0
    proposal_total = 0

    for sample in samples:
        oracle = sample.get("proper_terms") or {}
        if not isinstance(oracle, dict):
            continue
        extracted = sample.get("gpt_extracted_terms") or []
        proposed = gpt_terms_for_sample(sample)

        extracted_keys = {normalize_key(term) for term in extracted if term}
        for src in oracle:
            if not src:
                continue
            extraction_total += 1
            if normalize_key(src) in extracted_keys:
                extraction_hits += 1

        for src, oracle_tgt in oracle.items():
            if not src or not oracle_tgt:
                continue
            src_key = normalize_key(src)
            matched_key = next(
                (prop_src for prop_src in proposed if normalize_key(prop_src) == src_key), None
            )
            if matched_key is None:
                continue
            proposal_total += 1
            if normalize_key(proposed[matched_key]) == normalize_key(oracle_tgt):
                proposal_hits += 1

    return {
        "extraction_overlap_pct": (
            extraction_hits / extraction_total * 100 if extraction_total else None
        ),
        "extraction_hits": extraction_hits,
        "extraction_total": extraction_total,
        "proposal_match_pct": proposal_hits / proposal_total * 100 if proposal_total else None,
        "proposal_hits": proposal_hits,
        "proposal_total": proposal_total,
    }


def fmt_metric(value: float | None, digits: int = 2) -> str:
    return "N/A" if value is None else f"{value:.{digits}f}"


def evaluate_language(lang: str, samples: list[dict[str, Any]], output_dir: Path) -> dict[str, Any]:
    config = LANG_CONFIG[lang]
    ref_field = config["ref_field"]

    pred_path = output_dir / f"{lang}_dev_v1_gpt_proposed_term_predictions.jsonl"
    save_jsonl(pred_path, samples)

    clean_preds = [s.get(f"prediction_{MODE}_clean", "") for s in samples]
    refs = [s.get(ref_field, "") for s in samples]

    metrics: dict[str, Any] = {}
    if samples and clean_preds:
        metrics.update(compute_bleu_chrf(clean_preds, refs))
        term_acc = terminology_accuracy_gpt(clean_preds, samples)
        term_cons = terminology_consistency_gpt(clean_preds, samples)
        metrics["terminology_accuracy"] = term_acc
        metrics["terminology_consistency"] = term_cons
        metrics["oracle_diagnostics"] = compute_oracle_diagnostics(samples)
        diag = metrics["oracle_diagnostics"]
        print(
            f"[{lang}/{MODE}] BLEU={fmt_metric(metrics['bleu'])} "
            f"chrF={fmt_metric(metrics['chrf'])} "
            f"term_acc={fmt_metric(term_acc['avg_ratio_pct'])}% "
            f"extract_overlap={fmt_metric(diag['extraction_overlap_pct'])}% "
            f"proposal_match={fmt_metric(diag['proposal_match_pct'])}%"
        )

    return {"predictions_file": repo_rel_path(pred_path), "metrics": metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "pipeline": "gpt_extract_propose_translate",
        "cache_dir": repo_rel_path(args.cache_dir),
        "provider": "openrouter",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openai/gpt-4o-mini",
        "modes": [MODE],
        "languages": {},
    }

    for lang in LANG_GROUPS:
        path = cache_path(args.cache_dir, lang)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cached GPT-proposed data for {lang}: {path}")
        samples = load_jsonl(path)
        config = LANG_CONFIG[lang]
        print(f"\n=== {lang}: {len(samples)} samples -> {config['target_lang']} ===")
        result = evaluate_language(lang, samples, args.output_dir)
        summary["languages"][lang] = {
            "data_file": repo_rel_path(path),
            "sample_count": len(samples),
            **config,
            "modes": {MODE: result},
        }

    metrics_path = args.output_dir / "metrics_summary.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nDone.", metrics_path)


if __name__ == "__main__":
    main()
