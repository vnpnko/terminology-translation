# Best LoRA vs GPT

Compares the best LoRA configuration (`qwen_lora_7b_zero_shot_2_epochs`, see note below) against GPT-4o-mini, `proper_term` mode only. Reuses the parent [`lora_finetuning/`](../README.md)'s shared `shared/results/`, `shared/data/`, and `shared/run_registry.json` — see that README for run naming and registered-runs details.

## Report table

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/best_models.xlsx` | Best LoRA config vs GPT-4o-mini; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/lora_finetuning/best_models/scripts/compare_best_models_to_excel.py` — LoRA run selectable via `--lora-run` (default `lora_2_epoch_zero_shot`) |

**Note:** defaults to `lora_2_epoch_zero_shot` (override with `--lora-run`). This follows the evidence in [`epoch_ablation/report/epoch_ablation.xlsx`](../epoch_ablation/report/epoch_ablation.xlsx): going from 2 to 3 epochs drops training loss much further (~0.05–0.07 vs ~0.14–0.17) while held-out BLEU/chrF plateau or slightly regress and only Term Accuracy keeps rising — an overfitting signature, so 2 epochs is treated as the better generalizing checkpoint.

## Output

[`figures/fig_lora_best_models.pdf`](figures/fig_lora_best_models.pdf) — best LoRA config (Qwen 7B, 2 epochs, zero-shot) vs. GPT-4o-mini (few-shot), by language pair; copied to [`poster/figures/`](../../../poster/figures/fig_lora_best_models.pdf) for the poster.

## Scripts

| File | Role |
|------|------|
| `scripts/compare_best_models_to_excel.py` | Builds `report/best_models.xlsx` from `metrics_summary.json`; LoRA run overridable via `--lora-run`; shared loading/extraction helpers from [`../shared/scripts/compare_common.py`](../shared/scripts/compare_common.py) |
| `scripts/figure_best_models.py` | Builds the poster's `fig_lora_best_models` figure (`build_best_models_figure`; run via `python shared/lib/analysis/generate_result_figures.py --only best_models`); resolves the LoRA run via [`../shared/run_registry.json`](../shared/run_registry.json) and shared loading/extraction helpers from [`../shared/scripts/compare_common.py`](../shared/scripts/compare_common.py) |
