#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.mt_eval.core import (  # noqa: E402
    build_translation_prompt,
    compute_bleu_chrf,
    load_jsonl,
    output_tag_for_language,
    save_jsonl,
    strip_output_tags,
    terminology_accuracy_advanced,
    terminology_consistency_advanced,
    terminology_for_mode,
)


OPENAI_MODEL = "openai/gpt-4o-mini"
DEFAULT_MODES = ["no_term", "proper_term", "random_term"]
TARGET_LANG = "German"
REF_FIELD = "de"
OUTPUT_DIR = REPO_ROOT / "Baseline" / "openrouter_outputs"
ENV_FILE = REPO_ROOT / "scripts" / ".env"

_print_lock = threading.Lock()


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


def log(*args: Any, **kwargs: Any) -> None:
    with _print_lock:
        print(*args, **kwargs)


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


def translate_sample(
    client: OpenAI,
    sample_en: str,
    terminology: dict[str, str] | None = None,
    target_lang: str = TARGET_LANG,
    max_new_tokens: int = 256,
    retries: int = 3,
    sleep_seconds: int = 2,
) -> str:
    prompt = build_translation_prompt(sample_en, target_lang, terminology)

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a translation assistant. "
                            "You only translate English to German."
                        ),
                    },
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                max_tokens=max_new_tokens,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            last_error = exc
            print(f"API error on attempt {attempt}/{retries}: {exc}")
            if attempt < retries:
                time.sleep(sleep_seconds)

    raise RuntimeError(f"Translation failed after {retries} attempts: {last_error}")


def run_mode_evaluation(
    mode: str,
    samples: list[dict[str, Any]],
    client: OpenAI,
    base_output_dir: Path,
    data_stem: str,
    tqdm_position: int,
) -> dict[str, Any]:
    log(f"\n--- Mode: {mode} (started) ---")

    preds: list[str] = []
    records: list[dict[str, Any]] = []

    for sample in tqdm(
        samples,
        desc=f"{data_stem} - {mode}",
        position=tqdm_position,
        leave=True,
    ):
        pred = translate_sample(
            client,
            sample_en=sample.get("en", ""),
            terminology=terminology_for_mode(sample, mode),
            target_lang=TARGET_LANG,
        )
        preds.append(pred)

        record = sample.copy()
        record[f"prediction_{mode}"] = pred
        record[f"prediction_{mode}_clean"] = strip_output_tags(pred)
        records.append(record)

    clean_preds = [strip_output_tags(pred) for pred in preds]
    output_jsonl = base_output_dir / f"{data_stem}_{mode}_predictions.jsonl"
    save_jsonl(output_jsonl, records)
    log(f"[{mode}] Saved predictions to: {output_jsonl}")

    metrics: dict[str, Any] = {}

    if samples and REF_FIELD in samples[0]:
        refs = [sample.get(REF_FIELD, "") for sample in samples]
        metrics.update(compute_bleu_chrf(clean_preds, refs))

        term_mode = term_eval_mode(mode)
        term_acc = terminology_accuracy_advanced(samples, clean_preds, mode=term_mode)
        term_cons = terminology_consistency_advanced(samples, clean_preds, mode=term_mode)

        metrics["terminology_accuracy"] = term_acc
        metrics["terminology_consistency"] = term_cons

        log(f"[{mode}] BLEU: {fmt_metric(metrics['bleu'])}")
        log(f"[{mode}] chrF2++: {fmt_metric(metrics['chrf'])}")
        log(
            f"[{mode}] Terminology accuracy ratio %: "
            f"{fmt_metric(term_acc.get('avg_ratio_pct'))}"
        )
        log(f"[{mode}] Terminology terms counted: {term_acc.get('total_terms', 0)}")
        log(
            f"[{mode}] Macro-avg consistency: "
            f"{fmt_metric(term_cons.get('macro_avg_consistency'))}"
        )
        log(
            f"[{mode}] Weighted-avg consistency: "
            f"{fmt_metric(term_cons.get('weighted_avg_consistency'))}"
        )
    else:
        log(f"[{mode}] Skipping BLEU/chrF - reference field '{REF_FIELD}' not found.")

    log(f"--- Mode: {mode} (finished) ---")
    return {
        "predictions_file": str(output_jsonl),
        "metrics": metrics,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OpenRouter MT evaluation.")
    parser.add_argument(
        "--data-file",
        type=Path,
        default=REPO_ROOT / "Baseline" / "ende_dev_v2.jsonl",
    )
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES)
    parser.add_argument("--api-key", default=os.environ.get("OPENROUTER_API_KEY", ""))
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    data_file = args.data_file.resolve()
    if not data_file.is_file():
        raise FileNotFoundError(f"File not found: {data_file}")

    api_key = args.api_key or os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise EnvironmentError(
            f"OPENROUTER_API_KEY not set. Add it to {ENV_FILE} "
            "(copy from scripts/.env.example) or pass --api-key."
        )

    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    samples = load_jsonl(data_file)
    data_stem = data_file.stem
    base_output_dir = OUTPUT_DIR
    base_output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== File: {data_file}")
    print("=== Translation direction: English -> German")
    print(f"=== Samples: {len(samples)}")
    print(f"=== Model: {OPENAI_MODEL}")
    print(f"=== Output tag: {output_tag_for_language(TARGET_LANG)}")
    print(f"=== Modes (parallel, max_workers={len(args.modes)}): {args.modes}")

    all_results: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=len(args.modes)) as executor:
        futures = {
            executor.submit(
                run_mode_evaluation,
                mode,
                samples,
                client,
                base_output_dir,
                data_stem,
                args.modes.index(mode),
            ): mode
            for mode in args.modes
        }
        for future in as_completed(futures):
            mode = futures[future]
            all_results[mode] = future.result()

    metrics_path = base_output_dir / f"{data_stem}_metrics_summary.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print("\nEvaluation finished.")
    print("Saved metrics summary to:", metrics_path)


if __name__ == "__main__":
    main()

