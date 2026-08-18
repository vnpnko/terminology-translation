# 05 · GPT-proposed terms vs oracle dictionary

Compares GPT-4o-mini's `proper_term` mode (fed the oracle dictionary) against `gpt_proposed_term` mode (GPT extracts and translates its own terminology, with no access to the reference translation) on the `dev_v1/original` 500-line eval set, for all three language pairs. Both runs are zero-shot, so the comparison is apples-to-apples.

This experiment restores a comparison that previously existed (`results/dev_v1/original/gpt_pipeline/metrics_summary.json`, `report/modes/dev_v1_original_gpt_pipeline_mode_comparison.xlsx`) but was deleted during a 2026-08-15 refactor. It reuses the already-cached GPT pipeline output — no new API calls are made.

## Reproduce

```bash
# 1. Score the cached predictions (BLEU/chrF/terminology accuracy+consistency, oracle diagnostics)
python experiments/05_gpt_proposed_terms/scripts/compute_gpt_proposed_metrics.py

# 2. Build the comparison workbook
python experiments/05_gpt_proposed_terms/scripts/compare_gpt_pipeline_modes_to_excel.py
```

## Inputs

| Path | Description |
| ---- | ----------- |
| `data/dev_v1/dev_v1_gpt_proposed/{lang}_dev_v1_gpt_terms.jsonl` | Cached `gpt_extracted_terms`, `gpt_proposed_terms`, and `prediction_gpt_proposed_term(_clean)` per line — the output of an earlier GPT extract → propose → translate run |
| `results/dev_v1/original/zero_shot/gpt/metrics_summary.json` | The zero-shot `proper_term` (oracle) baseline metrics |

## Outputs

| Path | Produced by |
| ---- | ----------- |
| `results/dev_v1/original/zero_shot/gpt_pipeline/metrics_summary.json` | `compute_gpt_proposed_metrics.py` — BLEU, chrF, terminology accuracy/consistency (keyed off GPT's own proposed terms), and `oracle_diagnostics` (how well GPT's extraction/proposal matched the oracle dictionary) |
| `results/dev_v1/original/zero_shot/gpt_pipeline/{lang}/{lang}_dev_v1_gpt_proposed_term_predictions.jsonl` | `compute_gpt_proposed_metrics.py` — per-language predictions, mirroring the layout of the other modes |
| `report/dev_v1_original_gpt_pipeline_mode_comparison.xlsx` | `compare_gpt_pipeline_modes_to_excel.py` — `modes` sheet (proper_term vs gpt_proposed_term, best-metric highlighted) and `oracle_diagnostics` sheet |

## Comparability caveat

The cached `gpt_proposed_term` run has no `use_few_shot` field, i.e. it was run zero-shot. The current `proper_term` baseline has both a `zero_shot/gpt/` and a `few_shot/gpt/` variant; only the **zero-shot** one (`results/dev_v1/original/zero_shot/gpt/metrics_summary.json`) is a fair comparison partner — the few-shot variant uses a different prompting setup and would not isolate the effect of oracle vs. self-proposed terminology.
