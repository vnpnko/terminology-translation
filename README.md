# Terminology Translation

Experiments on terminology-constrained machine translation for WMT-style sentence-level data.

Each JSONL record has an English source sentence, a target-language reference, domain terminology, and random control terminology:

```json
{"en": "...", "de": "...", "proper_terms": {"term": "translation"}, "random_terms": {"word": "translation"}}
```

## Project Layout

```text
terminology-translation/
├── baselines/
├── scripts/
│   ├── core.py
│   ├── evaluation/
│   └── data_preparation/
├── data/
│   ├── sap_dev/
│   └── term_pairs/
└── reference/
    ├── shared_task_docs/
    ├── shared_task_track1/
    └── shared_task_track2/
```

## What To Run

For the OpenRouter/GPT baseline:

```bash
python scripts/evaluation/run_openrouter_gpt4o_mini_eval.py --data-file baselines/ende_dev_v2.sample.jsonl
```

For the local Qwen baseline:

```bash
python scripts/evaluation/run_local_qwen_eval.py --data-dir data/sap_dev
```

## Script Guide

| Script | Purpose |
|---|---|
| `scripts/evaluation/run_openrouter_gpt4o_mini_eval.py` | Runs OpenRouter `openai/gpt-4o-mini` translation evaluation over `no_term`, `proper_term`, and `random_term` modes. |
| `scripts/evaluation/run_local_qwen_eval.py` | Runs a local Hugging Face Qwen model evaluation with the same modes and metrics. |
| `scripts/data_preparation/annotate_proper_terms_openrouter.py` | Adds `proper_terms` to sentence pairs using an OpenRouter LLM. |
| `scripts/data_preparation/annotate_random_terms_openrouter.py` | Adds non-domain `random_terms` after `proper_terms` exist. |
| `scripts/data_preparation/fill_missing_translations_openrouter.py` | Fills missing target sentences and term translations from English records. |
| `scripts/data_preparation/collect_term_pairs_from_jsonl.py` | Extracts merged EN→target proper-term pairs from JSONL datasets. |
| `scripts/data_preparation/extract_english_proper_term_list.py` | Extracts English proper-term keys into a plain text list. |
| `scripts/data_preparation/filter_english_term_list.py` | Filters an English term list using stopwords, blocklists, and optional frequency filtering. |
| `scripts/data_preparation/strip_target_translations.py` | Clears target translations while keeping English and term keys. |
| `scripts/data_preparation/convert_term_postedits_to_jsonl.py` | Converts raw ASAP `term_postedits` flat files into this repo's JSONL format. |

## Shared Evaluation Code

`scripts/core.py` contains the reusable evaluation pieces:

- JSONL loading/saving
- prompt construction
- terminology mode selection
- output-tag stripping
- BLEU and chrF
- terminology accuracy
- terminology consistency

## Data And Reference

| Folder | Purpose |
|---|---|
| `baselines/` | Baseline notebooks plus the small German sample JSONL used by the OpenRouter runner. |
| `data/sap_dev/` | Original SAP/ASAP development JSONL files. |
| `data/term_pairs/` | Extracted proper-term pair lists used by data-preparation scripts. |
| `reference/shared_task_docs/` | WMT/shared-task reference documents. |
| `reference/shared_task_track1/` | Track 1 shared-task JSONL files. |
| `reference/shared_task_track2/` | Track 2 shared-task JSONL files. |

## Dependencies

OpenRouter evaluator:

```bash
pip install openai sacrebleu tqdm
```

Local Qwen evaluator:

```bash
pip install transformers accelerate torch sacrebleu tqdm
```
