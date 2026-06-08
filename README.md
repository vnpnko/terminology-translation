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

| Script                                 | Purpose                                                                                                                          |
| -------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| `analysis/compare_metrics_to_excel.py` | Compares GPT and Qwen `metrics_summary.json` files into a formatted Excel baseline comparison. Requires `pandas` and `openpyxl`. |

## Results

All results live in `results/`

Each `gpt/` and `qwen/` results directory contains a `metrics_summary.json` file.

## Report

Generated baseline comparison Excel files are written to `report/`.

## Data

All data lives in `data/`

| Path               | Description                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dev_v1/original/` | Original course dev set.                                                                                                                                           |
| `dev_v1/expand/`   | Expanded version of `original/` with additional `proper_terms`.                                                                                                    |
| `dev_v1/cleaned/`  | Cleaned version of `expand/` with terminology-poor `proper_terms` removed.                                                                                         |
| `dev_v2/`          | Dev set prepared from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus. |
