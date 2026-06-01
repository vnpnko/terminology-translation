#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path
from typing import Any

from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mt_eval.core import (  # noqa: E402
    build_translation_prompt,
    compute_bleu_chrf,
    load_jsonl,
    save_jsonl,
    strip_output_tags,
    terminology_accuracy_advanced,
    terminology_consistency_advanced,
    terminology_for_mode,
)


MODEL_NAME = "Qwen/Qwen2.5-Coder-3B-Instruct"
DEFAULT_MODES = ["no_term", "proper_term", "random_term"]
OUTPUT_DIR = REPO_ROOT / "Baseline" / "qwen_outputs"
SYSTEM_MESSAGE = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."


def detect_target_language(samples: list[dict[str, Any]]) -> tuple[str, str, str, str]:
    for sample in samples:
        if "de" in sample:
            return "de", "German", "deu", "de"
        if "es" in sample:
            return "es", "Spanish", "es", "es"
        if "ru" in sample:
            return "ru", "Russian", "ru", "ru"
    return "de", "German", "deu", "de"


def fmt_metric(value: float | None, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


def term_eval_mode(mode: str) -> str:
    if mode == "proper_term":
        return "proper_term"
    if mode == "random_term":
        return "random_term"
    return "no_term"


def load_model(model_name: str = MODEL_NAME):
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
    hf_logging.disable_progress_bar()

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def translate_sample(
    model,
    tokenizer,
    sample_en: str,
    terminology: dict[str, str] | None = None,
    target_lang: str = "German",
    max_new_tokens: int = 256,
) -> str:
    prompt = build_translation_prompt(sample_en, target_lang, terminology)
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
    )
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]


def run_mode_evaluation(
    mode: str,
    samples: list[dict[str, Any]],
    model,
    tokenizer,
    target_lang: str,
    ref_field: str,
    output_dir: Path,
    data_stem: str,
) -> dict[str, Any]:
    print(f"\n--- Mode: {mode} ---")
    preds: list[str] = []
    records: list[dict[str, Any]] = []

    for sample in tqdm(samples, desc=f"{data_stem} - {mode}"):
        pred = translate_sample(
            model,
            tokenizer,
            sample.get("en", ""),
            terminology=terminology_for_mode(sample, mode),
            target_lang=target_lang,
        )
        preds.append(pred)

        record = sample.copy()
        record[f"prediction_{mode}"] = pred
        record[f"prediction_{mode}_clean"] = strip_output_tags(pred)
        records.append(record)

    clean_preds = [strip_output_tags(pred) for pred in preds]
    output_jsonl = output_dir / f"{data_stem}_{mode}_predictions.jsonl"
    save_jsonl(output_jsonl, records)

    refs = [sample.get(ref_field) for sample in samples]
    metrics = compute_bleu_chrf(clean_preds, refs)

    term_mode = term_eval_mode(mode)
    term_acc = terminology_accuracy_advanced(samples, clean_preds, mode=term_mode)
    term_cons = terminology_consistency_advanced(samples, clean_preds, mode=term_mode)

    metrics["terminology_accuracy"] = term_acc
    metrics["terminology_consistency"] = term_cons

    print(f"BLEU: {fmt_metric(metrics['bleu'])}")
    print(f"chrF2++: {fmt_metric(metrics['chrf'])}")
    print(f"Terminology accuracy (ratio %): {fmt_metric(term_acc.get('avg_ratio_pct'))}")
    print(f"Terminology terms counted: {term_acc.get('total_terms', 0)}")
    print(f"Macro-avg consistency: {fmt_metric(term_cons.get('macro_avg_consistency'))}")
    print(f"Weighted-avg consistency: {fmt_metric(term_cons.get('weighted_avg_consistency'))}")

    return {
        "predictions_file": str(output_jsonl),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run local Qwen MT evaluation.")
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(
        path
        for path in glob.glob(str(data_dir / "*.jsonl"))
        if path.endswith("_dev.jsonl")
    )
    print("Found data files:", files)

    model, tokenizer = load_model(args.model_name)
    print(f"Model and tokenizer loaded: {args.model_name}")

    for filepath in files:
        samples = load_jsonl(filepath)
        _lang_code, lang_name, output_tag, ref_field = detect_target_language(samples)
        data_stem = Path(filepath).stem

        print(
            f"\n=== File: {filepath} | target={lang_name} "
            f"| tag={output_tag} "
            f"| samples={len(samples)} ==="
        )

        all_results: dict[str, Any] = {}
        for mode in args.modes:
            all_results[mode] = run_mode_evaluation(
                mode,
                samples,
                model,
                tokenizer,
                lang_name,
                ref_field,
                output_dir,
                data_stem,
            )

        metrics_path = output_dir / f"{data_stem}_metrics_summary.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print("Saved metrics summary to:", metrics_path)

    print("\nEvaluation finished.")


if __name__ == "__main__":
    main()
