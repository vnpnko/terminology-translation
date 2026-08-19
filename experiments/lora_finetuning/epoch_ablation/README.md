# LoRA epoch ablation

Compares 1 / 2 / 3 LoRA fine-tuning epochs (zero-shot), one sheet per model size (3B/7B), `proper_term` mode only. Reuses the parent [`lora_finetuning/`](../README.md)'s shared `results/`, `data/`, and `run_registry.json` — see that README for run naming and registered-runs details.

## Report table

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/epoch_ablation.xlsx` | 1 / 2 / 3 LoRA epochs (zero-shot), one sheet per model size; each cell colored red/yellow/green by ranking that (language, metric) value across the 3 epochs (worst/mid/best) | `python experiments/lora_finetuning/epoch_ablation/scripts/compare_epochs_to_excel.py` |

Feeds the epoch line-chart panels in the parent's [`fig_lora_finetuning`](../figures/fig_lora_finetuning.pdf) figure (not code-coupled — the figure reads `metrics_summary.json` independently, but visualizes the same underlying epoch progression).

## Scripts

| File | Role |
|------|------|
| `scripts/compare_epochs_to_excel.py` | Builds `report/epoch_ablation.xlsx` from `metrics_summary.json`, run selection driven by [`../run_registry.json`](../run_registry.json) |
