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

All scripts live in `scripts/data_preparation/`

| Script                                    | Purpose                                                                      |
| ----------------------------------------- | ---------------------------------------------------------------------------- |
| `convert_term_postedits_to_jsonl.py`      | Converts raw ASAP `term_postedits` flat files into JSONL mt-task format.     |
| `fill_missing_translations_openrouter.py` | Fills missing target sentences and term translations via OpenRouter.         |
| `annotate_proper_terms_openrouter.py`     | Adds domain `proper_terms` (1–2 IT terms per sentence) via OpenRouter.       |
| `annotate_random_terms_openrouter.py`     | Adds control `random_terms` (non-domain word pairs) after `proper_terms`.    |
| `clean_poor_proper_terms.py`              | Removes weak or generic entries from `proper_terms`.                         |
| `expand_terms.py`                         | Appends additional term pairs to `proper_terms`.                             |
| `collect_term_pairs_from_jsonl.py`        | Aggregates `proper_terms` from v1/v2 dev files into unique term-pair JSONL.  |
| `strip_target_translations.py`            | Clears target sentences and term values while keeping English and term keys. |

## Results

All results live in `results/`

Each `gpt/` and `qwen/` results directory contains a `metrics_summary.json` file. This file reports key evaluation metrics for the translations, including:

- `bleu`: BLEU score
- `chrf`: ChrF score
- `terminology_accuracy`: Metrics for accuracy of required terminology usage
- `terminology_consistency`: Metrics for consistency of terminology use

## Data

All data lives in `data/`

| Path               | Description                                                                                                                                                        |
| ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `dev_v1/original/` | Original course dev set.                                                                                                                                           |
| `dev_v1/expand/`   | Expanded version of `original/` with additional `proper_terms`.                                                                                                    |
| `dev_v1/cleaned/`  | Cleaned version of `expand/` with terminology-poor `proper_terms` removed.                                                                                         |
| `dev_v2/`          | Dev set prepared from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus. |
