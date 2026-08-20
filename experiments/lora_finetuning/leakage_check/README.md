# Data-leakage honesty check

Checks whether LoRA training on `dev_v2` leaked into the `dev_v1` test set: splits `dev_v1` test sentences into `overlap`/`no_overlap` subsets by ≥50% token containment with the training set, then compares `overlap_data` vs. `no_overlap_data` performance for `qwen_base` (untrained, control), `gpt` (closed model, never exposed to `dev_v2` training data, control), and `qwen_lora` (trained, the model under test) — controls first so the overlap-vs-no-overlap gap visibly grows with training exposure. `proper_term` mode only.

`data/{overlap,no_overlap}/` (the test-sentence split) lives here since this check is its only consumer; it's built from the parent [`lora_finetuning/`](../README.md)'s shared `shared/data/training/` and `shared/data/test/` via `filter_test_sentence_overlap.py` below. The resulting `results/*/test_cleaned_by_sentences/{overlap,no_overlap}/metrics_summary.json` files (produced by re-running the model notebooks against this split) live under the parent's shared `shared/results/` tree, alongside each model's regular results.

## Report table

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/leakage_honesty_check.xlsx` | Single sheet `overlap_vs_no_overlap_data` with `overlap_data`/`no_overlap_data` column groups (named for the split criterion, not a "bad"/"good" judgment) and 3 stacked 3-row model blocks (`qwen_base`, `gpt`, `qwen_lora`); each cell colored green (higher/tied-highest) or yellow (lower) against its overlap_data/no_overlap_data counterpart, no red | `python experiments/lora_finetuning/leakage_check/scripts/compare_leakage_honesty_check_to_excel.py` — LoRA run selectable via `--lora-run` (default `lora_2_epoch_zero_shot`) |

**Note:** fully script-generated, with column groups named `overlap_data`/`no_overlap_data` for the actual split criterion (≥50% token containment with the training set). The `test_cleaned_by_sentences/no_overlap` split it reads for `qwen_base` is a real, verified no-overlap subset.

**Note:** this check uses only sentence-level overlap filtering. Term-level overlap filtering (splitting on shared *term* overlap rather than whole-sentence overlap) isn't used: almost every `dev_v1` sentence shares a term-dictionary entry with `dev_v2` training data, which would produce a heavily skewed split (11–14 "good" sentences vs. 483–486 "bad" per language) — not a statistically meaningful comparison.

**Note:** only run for Qwen **7B**, not 3B — deliberately: the check's purpose is to compare the untrained base model against the trained (LoRA) model on the same architecture to see whether leakage is present, not to compare across model sizes, so one model size is sufficient.

**Model names:** the workbook's `model` column uses short labels; here's what each one is concretely: `gpt` = GPT-4o-mini; `qwen_base` = Qwen2.5-7B, base model, few-shot (`run_registry.json`'s `base_few_shot`, folder `qwen_base` — the folder name itself doesn't indicate few-shot, but the run is); `qwen_lora` = Qwen2.5-7B, LoRA fine-tuned, 2 epochs, zero-shot (`run_registry.json`'s `lora_2_epoch_zero_shot`, folder `qwen_lora_no_few_shots_2_epochs`, the script's `--lora-run` default).

## Scripts

| File | Role |
|------|------|
| `scripts/compare_leakage_honesty_check_to_excel.py` | Builds `report/leakage_honesty_check.xlsx` from `metrics_summary.json` under `.../test_cleaned_by_sentences/{overlap,no_overlap}/`; LoRA run overridable via `--lora-run`; shared loading/extraction helpers from [`../shared/scripts/compare_common.py`](../shared/scripts/compare_common.py) |
| `scripts/filter_test_sentence_overlap.py` | Splits `dev_v1` test sentences into `overlap`/`no_overlap` subsets by ≥50% token containment with the training set, balanced to the minimum count across languages; writes `data/{overlap,no_overlap}/{lang}_dev_v1_test.jsonl` |
