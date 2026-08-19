# Known gaps and open questions

Things about the current repo state worth knowing before writing the paper. Companion to [`README.md`](README.md).

## Script/data inconsistencies

- `experiments/04_lora_finetuning/scripts/filter_test_sentence_overlap.py`'s default output dir (`data/test_cleaned_gpt`) doesn't match the committed `data/test_cleaned_by_sentences/` — regenerating needs `--output-dir` passed explicitly.
- `experiments/04_lora_finetuning/report/best_models.xlsx` uses openpyxl's default `Sheet1` name, unlike its descriptively-named siblings (`epoch_ablation.xlsx`'s `Qwen2.5-3B`/`Qwen2.5-7B`, `leakage_honesty_check.xlsx`'s `overlap_vs_no_overlap_data`, etc.) — cosmetic only.
