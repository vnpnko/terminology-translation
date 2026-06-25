# Terminology Translation

Experiments on terminology-constrained machine translation for WMT-style sentence-level data.

Each JSONL record has an English source sentence, a target-language reference, domain terminology, and random control terminology:

```json
{
  "en": "...",
  "de": "...",
  "proper_terms": [{ "term": "translation" }],
  "random_terms": [{ "word": "translation" }]
}
```

## Scripts

Scripts live in `scripts/`.

### Data preparation

| Script                                                     | Purpose                                                                      |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `data_preparation/convert_term_postedits_to_jsonl.py`      | Converts raw ASAP `term_postedits` flat files into JSONL mt-task format.     |
| `data_preparation/fill_missing_translations_openrouter.py` | Fills missing target sentences and term translations via OpenRouter.         |
| `data_preparation/annotate_proper_terms_openrouter.py`     | Adds domain `proper_terms` (1–2 IT terms per sentence) via OpenRouter.       |
| `data_preparation/annotate_random_terms_openrouter.py`     | Adds control `random_terms` (non-domain word pairs) after `proper_terms`.    |
| `data_preparation/clean_poor_proper_terms.py`              | Removes weak or generic entries from `proper_terms`.                         |
| `data_preparation/expand_terms.py`                         | Appends additional term pairs to `proper_terms`.                             |
| `data_preparation/collect_term_pairs_from_jsonl.py`        | Aggregates `proper_terms` from v1/v2 dev files into unique term-pair JSONL.  |
| `data_preparation/build_term_dictionary.py`                | Builds provenance-aware term dictionary from dev_v2 (GPT via OpenRouter).    |
| `data_preparation/apply_dictionary_to_dev_v1.py`             | Applies dictionary to dev_v1 into `dev_v1/dictionary/` (optional, new files).  |
| `data_preparation/strip_target_translations.py`            | Clears target sentences and term values while keeping English and term keys. |

### GPT term pipeline (500-sentence eval)

| Script | Purpose |
| ------ | ------- |
| `run_gpt_term_pipeline.py` | 3-step GPT pipeline: extract EN terms → propose target translations (no reference) → translate. |

This is an **inference-time experiment**, not a dictionary build. Reference translations (`de`/`es`/`ru`) in the JSONL are used **for evaluation only** — they are never sent to GPT during steps 1–2.

```bash
# Pilot (10 lines, German)
python scripts/run_gpt_term_pipeline.py --lang ende --limit 10

# Full run (500 lines × 3 language pairs)
python scripts/run_gpt_term_pipeline.py --all

# Resume after interruption
python scripts/run_gpt_term_pipeline.py --all --resume

# Compare oracle vs GPT-proposed modes
python scripts/analysis/compare_gpt_pipeline_modes.py
```

**Outputs**

| Path | Description |
| ---- | ----------- |
| `data/dev_v1/gpt_proposed/` | Cached `gpt_extracted_terms` and `gpt_proposed_terms` per line |
| `results/dev_v1/original/gpt_pipeline/` | Translations + `metrics_summary.json` |

**Co-editor handoff (Qwen):** load `data/dev_v1/gpt_proposed/{lang}_dev_v1_gpt_terms.jsonl`, use `gpt_proposed_terms` as the terminology dict in the Qwen baseline notebooks (translation step only). Write results to e.g. `results/dev_v1/original/qwen_3b_gpt_terms/`.

### Analysis

All analysis scripts require `pandas` and `openpyxl`.

| Script                                  | Purpose                                                                      |
| --------------------------------------- | ---------------------------------------------------------------------------- |
| `analysis/compare_models_to_excel.py`   | Compares GPT, Qwen 3B, and Qwen 7B. Rows grouped by mode.                    |
| `analysis/compare_modes_to_excel.py`    | Compares `no_term`, `proper_term`, and `random_term`. Rows grouped by model. |
| `analysis/compare_datasets_to_excel.py` | Compares `dev_v1/original` vs `dev_v2`. Rows grouped by mode.                |
| `analysis/compare_gpt_pipeline_modes.py` | Compares GPT baseline modes vs `gpt_proposed_term` pipeline.                  |

Example commands:

## Results

All results live in `results/`

Each `gpt/`, `qwen_3b/` and `qwen_7b/` results directory contains a `metrics_summary.json` file.

## Report

Generated comparison Excel files are written under `report/`:

| Directory          | Produced by                    | Naming pattern                                                                   |
| ------------------ | ------------------------------ | -------------------------------------------------------------------------------- |
| `report/models/`   | `compare_models_to_excel.py`   | `<dataset>_model_comparison.xlsx` (e.g. `dev_v1_original_model_comparison.xlsx`) |
| `report/modes/`    | `compare_modes_to_excel.py`    | `<dataset>_mode_comparison.xlsx`                                                 |
| `report/modes/`    | `compare_gpt_pipeline_modes.py` | `dev_v1_original_gpt_pipeline_mode_comparison.xlsx`                              |
| `report/datasets/` | `compare_datasets_to_excel.py` | `dev_v1_original_vs_dev_v2_<model>_dataset_comparison.xlsx`                      |

## Data

All data lives in `data/`

| Path               | Description                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dev_v1/original/` | Original course dev set.                                                                                                                                           |
| `dev_v1/expand/`   | Expanded version of `original/` with additional `proper_terms`.                                                                                                    |
| `dev_v1/cleaned/`  | Cleaned version of `expand/` with terminology-poor `proper_terms` removed.                                                                                         |
| `dev_v1/dictionary/` | dev_v1 enriched from term dictionary (optional output of `apply_dictionary_to_dev_v1.py`).                                                                         |
| `dev_v1/gpt_proposed/` | GPT-extracted and GPT-proposed term pairs for the 500-line eval set (output of `run_gpt_term_pipeline.py`).                                                          |
| `dev_v2/`          | Dev set prepared from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus. |
| `term_dictionary/` | Term dictionary built from dev_v2 with line IDs, lemmas, and observed inflections.                                                                                  |
| `term_pairs/`      | Flat aggregated EN→target term-pair lists from v1/v2.                                                                                                              |

### Term dictionary pipeline

```bash
conda activate terminology-translation
pip install -r requirements.txt
python -m spacy download en_core_web_sm de_core_news_sm es_core_news_sm

# Build dictionary from dev_v2 (dry run)
python scripts/data_preparation/build_term_dictionary.py --all --limit 50

# Full build (requires OPENROUTER_API_KEY in .env)
python scripts/data_preparation/build_term_dictionary.py --all

# Optional: apply to dev_v1 (writes new files only)
python scripts/data_preparation/apply_dictionary_to_dev_v1.py --all
```
