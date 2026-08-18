# Known gaps and open questions

Things about the current repo state worth knowing before writing the paper. Companion to [`README.md`](README.md).

## Missing or unreproducible data

- `experiments/03_dataset_comparison/data/dev_v2_dictionary/` doesn't exist on disk. It's only needed to rebuild `data/dev_v1/dev_v1_dictionary/` from scratch (`build_term_dictionary.py --all`, then `apply_dictionary_to_dev_v1.py`, both requiring `OPENROUTER_API_KEY`); the already-built `data/dev_v1/dev_v1_dictionary/` and its results/report tables work today without it.
- `poster/figures/fig_dictionary_vs_original.pdf` exists but no script in this repo regenerates it. The closest current equivalent, `compare_v1_variants_to_excel.py`, produces an Excel comparison, not a figure.
- `experiments/04_lora_finetuning/scripts/remove_dev_v2_overlap.py` implements the dev_v2 exact-overlap-removal pipeline (2000 → 1500 lines/language, then a few more held back for few-shot examples), but neither `remove_overlap_report.json` nor `experiments/04_lora_finetuning/data/dev_v2_deduped/` exists in the current tree — the pipeline's numbers, as stated in `README.md`, aren't backed by a live, regeneratable artifact.
- `data/test_cleaned_by_terms/{data_good,data_bad}/` is prepared but never evaluated — no `results/.../test_cleaned_by_terms/` exists anywhere. The split is heavily skewed (11–14 "good" sentences vs. 483–486 "bad" per language pair — almost every `dev_v1` sentence shares a term-dictionary entry with `dev_v2` training data), so it isn't a statistically meaningful comparison even if run.
- `test_cleaned_by_sentences`'s good/bad leakage-honesty evaluation exists only for Qwen 7B (`experiments/04_lora_finetuning/results/Qwen2.5-7B/`); no 3B good/bad results exist.

## Duplicated / unclear-canonical data

- `data/dev_v2/dev_v2_training/*.jsonl` and `experiments/04_lora_finetuning/data/training/*.jsonl` hold the same derived, post-removal training data under two separate paths, with nothing in the repo indicating which is canonical.

## Confounds to state plainly in the paper

- The LoRA few-shot ablation is confounded with epoch count: `run_registry.json` has only one few-shot LoRA run (`lora_1_epoch_few_shot`); the 2- and 3-epoch runs are both zero-shot, so there's no clean way to isolate "few-shot effect" from "epoch count" on the LoRA side.
- The leakage-honesty check's `overlap_data`/`no_overlap_data` split is confounded with sentence length and terminology density, not just training-set overlap: `overlap_data` sentences are 30–49% shorter on average and have meaningfully higher terminology density (most pronounced on `enes`, 38.8% vs. 17.8%), while total unique terms per subset are nearly identical. This is why even GPT (never trained on `dev_v2`) shows a real gap between the two subsets — the leakage-inflation reading for `qwen_lora` should be stated as "on top of this difficulty confound," not as if the confound were absent. See `README.md` §3.4.2 for the full write-up and numbers.

## Script/data inconsistencies

- `experiments/04_lora_finetuning/scripts/filter_test_sentence_overlap.py`'s default output dir (`data/test_cleaned_gpt`) doesn't match the committed `data/test_cleaned_by_sentences/`; `filter_test_term_overlap.py`'s default test dir (`data/test_cleaned`) doesn't match the committed `data/test_cleaned_by_terms/`. Both need `--output-dir`/`--test-dir` passed explicitly to reproduce the current data.
- `filter_test_term_overlap.py` overwrites its input file in place (`write_jsonl(test_path, kept)`) rather than writing to a separate output file, unlike its sentence-level sibling — re-running it destructively modifies the source data.
- `experiments/04_lora_finetuning/scripts/run_registry.json`'s `workbook` fields, and `metrics_parser.py`/`sheet_builders.py`/`export_finetuning_report.py`, together implement an export pipeline for a `results_Qwen2.5-{3B,7B}.xlsx`/`results_GPT4o-mini.xlsx` workbook format that produces nothing currently used in `report/`.
- `experiments/04_lora_finetuning/report/best_models.xlsx` uses openpyxl's default `Sheet1` name, unlike its descriptively-named siblings (`epoch_ablation.xlsx`'s `Qwen2.5-3B`/`Qwen2.5-7B`, `leakage_honesty_check.xlsx`'s `overlap_vs_no_overlap_data`, etc.) — cosmetic only.

## Open questions for the user

1. **Few-shot scope for the paper**: report only the clean baseline-level few-shot comparison (GPT/Qwen 3B/7B, no fine-tuning) and describe the LoRA-side epoch/few-shot confound as a stated limitation, or invest time in running the missing `lora_2_epoch_few_shot`/`lora_3_epoch_few_shot` configs first for a cleaner ablation?
2. **`test_cleaned_by_terms`**: skip it and explain the class-imbalance reason in the paper (recommended, given the skew), or is there a reason to still want it run?
3. **Canonical training-set path**: `data/dev_v2/dev_v2_training/` vs. `experiments/04_lora_finetuning/data/training/` — which one is canonical, so the other can be removed or symlinked?
4. **Dead export pipeline**: delete `run_registry.json`'s unused `workbook` fields and the `export_finetuning_report.py`/`sheet_builders.py`/`metrics_parser.py` files outright, or leave them as reference?
