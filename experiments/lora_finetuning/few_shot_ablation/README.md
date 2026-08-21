# Zero-shot vs few-shot ablation

Compares `zero_shot` vs `few_shot` prompting, one sheet per model (GPT-4o-mini, Qwen 3B, Qwen 7B); the Qwen sheets stack a `qwen_base` (untrained) block and a `qwen_lora` (1-epoch fine-tuned) block. `proper_term` mode only. Reuses the parent [`lora_finetuning/`](../README.md)'s shared `shared/results/` and `shared/run_registry.json` (for the `qwen_lora` rows) — see that README for run naming and registered-runs details.

Note: only the `qwen_lora` block involves LoRA fine-tuning; the `GPT-4o-mini` sheet and each Qwen sheet's `qwen_base` block compare prompting strategy on **untrained** models. Kept together in one workbook (and one experiment) since the comparison axis — zero-shot vs. few-shot — is the same across all three blocks.

## Report table

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/zero_shot_vs_few_shot_ablation.xlsx` | `zero_shot` vs `few_shot`, one sheet per model; Qwen sheets stack `qwen_base` and `qwen_lora` rows; each cell colored green (higher) or red (lower), yellow if tied | `python experiments/lora_finetuning/few_shot_ablation/scripts/compare_few_shots_to_excel.py` — GPT/`qwen_base` rows split across two results trees (see script docstring); `qwen_lora` rows are fully local, driven by `../shared/run_registry.json` |

**Note:** fills in `Qwen2.5-7B`'s `qwen_base` `zero_shot` cells from `shared/results/dev_v1/original/zero_shot/qwen_7b/` (the shared baseline tree, not this experiment's local `shared/results/`).

## Scripts

| File | Role |
|------|------|
| `scripts/compare_few_shots_to_excel.py` | Builds `report/zero_shot_vs_few_shot_ablation.xlsx` from `metrics_summary.json` in both `shared/results/` (root) and `experiments/lora_finetuning/shared/results/`; shared loading/extraction helpers from [`../shared/scripts/compare_common.py`](../shared/scripts/compare_common.py) |
