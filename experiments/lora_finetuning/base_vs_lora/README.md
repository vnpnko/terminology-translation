# Base (few-shot) vs LoRA (1 epoch, zero-shot)

Compares `qwen_base_few_shot` vs `qwen_lora_zero_shot` (1 epoch), for 3B and 7B — does 1 epoch of fine-tuning beat few-shot prompting without any fine-tuning? `proper_term` mode only. Reuses the parent [`lora_finetuning/`](../README.md)'s shared `shared/results/`, `shared/data/`, and `shared/run_registry.json` — see that README for run naming and registered-runs details.

## Report table

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/base_vs_lora.xlsx` | `qwen_base_few_shot` vs `qwen_lora_zero_shot` (1 epoch), for 3B and 7B; each cell colored green (higher) or red (lower), yellow if tied | `python experiments/lora_finetuning/base_vs_lora/scripts/compare_base_vs_lora_to_excel.py` |

## Scripts

| File | Role |
|------|------|
| `scripts/compare_base_vs_lora_to_excel.py` | Builds `report/base_vs_lora.xlsx` from `metrics_summary.json`, run selection driven by [`../shared/run_registry.json`](../shared/run_registry.json); shared loading/extraction helpers from [`../shared/scripts/compare_common.py`](../shared/scripts/compare_common.py) |
