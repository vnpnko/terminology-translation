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
| `data_preparation/strip_target_translations.py`            | Clears target sentences and term values while keeping English and term keys. |

### Analysis

All analysis scripts require `pandas` and `openpyxl`.

| Script                                  | Purpose                                                                      |
| --------------------------------------- | ---------------------------------------------------------------------------- |
| `analysis/compare_models_to_excel.py`   | Compares GPT, Qwen 3B, and Qwen 7B. Rows grouped by mode.                    |
| `analysis/compare_modes_to_excel.py`    | Compares `no_term`, `proper_term`, and `random_term`. Rows grouped by model. |
| `analysis/compare_datasets_to_excel.py` | Compares `dev_v1/original` vs `dev_v2`. Rows grouped by mode.                |

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
| `report/datasets/` | `compare_datasets_to_excel.py` | `dev_v1_original_vs_dev_v2_<model>_dataset_comparison.xlsx`                      |

## Data

All data lives in `data/`

| Path               | Description                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dev_v1/original/` | Original course dev set.                                                                                                                                           |
| `dev_v1/expand/`   | Expanded version of `original/` with additional `proper_terms`.                                                                                                    |
| `dev_v1/cleaned/`  | Cleaned version of `expand/` with terminology-poor `proper_terms` removed.                                                                                         |
| `dev_v2/`          | Dev set prepared from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus. |
