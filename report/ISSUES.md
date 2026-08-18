# Known gaps and open questions

Things about the current repo state worth knowing before writing the paper. Companion to [`README.md`](README.md).

## Missing or unreproducible data

- `test_cleaned_by_sentences`'s good/bad leakage-honesty evaluation exists only for Qwen 7B (`experiments/04_lora_finetuning/results/Qwen2.5-7B/`); no 3B good/bad results exist.

## Confounds to state plainly in the paper

- The LoRA few-shot ablation is confounded with epoch count: `run_registry.json` has only one few-shot LoRA run (`lora_1_epoch_few_shot`); the 2- and 3-epoch runs are both zero-shot, so there's no clean way to isolate "few-shot effect" from "epoch count" on the LoRA side.
- The leakage-honesty check's `overlap_data`/`no_overlap_data` split is confounded with sentence length and terminology density, not just training-set overlap: `overlap_data` sentences are 30–49% shorter on average and have meaningfully higher terminology density (most pronounced on `enes`, 38.8% vs. 17.8%), while total unique terms per subset are nearly identical. This is why even GPT (never trained on `dev_v2`) shows a real gap between the two subsets — the leakage-inflation reading for `qwen_lora` should be stated as "on top of this difficulty confound," not as if the confound were absent. See `README.md` §3.4.2 for the full write-up and numbers.

## Script/data inconsistencies

- `experiments/04_lora_finetuning/scripts/filter_test_sentence_overlap.py`'s default output dir (`data/test_cleaned_gpt`) doesn't match the committed `data/test_cleaned_by_sentences/` — regenerating needs `--output-dir` passed explicitly.
- `experiments/04_lora_finetuning/scripts/run_registry.json`'s `workbook` fields, and `metrics_parser.py`/`sheet_builders.py`/`export_finetuning_report.py`, together implement an export pipeline for a `results_Qwen2.5-{3B,7B}.xlsx`/`results_GPT4o-mini.xlsx` workbook format that produces nothing currently used in `report/`.
- `experiments/04_lora_finetuning/report/best_models.xlsx` uses openpyxl's default `Sheet1` name, unlike its descriptively-named siblings (`epoch_ablation.xlsx`'s `Qwen2.5-3B`/`Qwen2.5-7B`, `leakage_honesty_check.xlsx`'s `overlap_vs_no_overlap_data`, etc.) — cosmetic only.

## Open questions for the user

1. **Few-shot scope for the paper**: report only the clean baseline-level few-shot comparison (GPT/Qwen 3B/7B, no fine-tuning) and describe the LoRA-side epoch/few-shot confound as a stated limitation, or invest time in running the missing `lora_2_epoch_few_shot`/`lora_3_epoch_few_shot` configs first for a cleaner ablation?
2. **Dead export pipeline**: delete `run_registry.json`'s unused `workbook` fields and the `export_finetuning_report.py`/`sheet_builders.py`/`metrics_parser.py` files outright, or leave them as reference?
