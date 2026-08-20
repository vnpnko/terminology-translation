# LoRA epoch ablation

Compares 1 / 2 / 3 LoRA fine-tuning epochs (zero-shot), one sheet per model size (3B/7B), `proper_term` mode only. Reuses the parent [`lora_finetuning/`](../README.md)'s shared `shared/results/`, `shared/data/`, and `shared/run_registry.json` — see that README for run naming and registered-runs details.

## Report table

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/epoch_ablation.xlsx` | 1 / 2 / 3 LoRA epochs (zero-shot), one sheet per model size; each cell colored red/yellow/green by ranking that (language, metric) value across the 3 epochs (worst/mid/best) | `python experiments/lora_finetuning/epoch_ablation/scripts/compare_epochs_to_excel.py` |

## Output

[`figures/fig_lora_epoch_ablation.pdf`](figures/fig_lora_epoch_ablation.pdf) — BLEU / term accuracy vs. LoRA epoch count (0=zero-shot base, 1/2/3=LoRA epochs), one line per model size (3B/7B); copied to [`poster/figures/`](../../../poster/figures/fig_lora_epoch_ablation.pdf) for the poster.

## Scripts

| File | Role |
|------|------|
| `scripts/compare_epochs_to_excel.py` | Builds `report/epoch_ablation.xlsx` from `metrics_summary.json`, run selection driven by [`../shared/run_registry.json`](../shared/run_registry.json); shared loading/extraction helpers from [`../shared/scripts/compare_common.py`](../shared/scripts/compare_common.py) |
| `scripts/figure_epoch_ablation.py` | Builds the poster's `fig_lora_epoch_ablation` figure (`build_epoch_ablation_figure`; run via `python shared/lib/analysis/generate_result_figures.py --only epoch_ablation`); imports [`../shared/scripts/metrics_parser.py`](../shared/scripts/metrics_parser.py) for `metrics_summary.json` loading/aggregation helpers |
