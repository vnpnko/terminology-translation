# Finetuning experiment Excel export

Generate supervisor-ready Excel workbooks from `metrics_summary.json` and `training_loss.txt` files under `Finetuning experiment/results/`.

## Quick start

From the repository root:

```bash
python "Finetuning experiment/scripts/export_finetuning_report.py"
```

Outputs land in `Finetuning experiment/report/`:

| File | Contents |
|------|----------|
| `results_Qwen2.5-3B.xlsx` | 3B runs: main results, base vs LoRA, epoch ablation, config, training loss |
| `results_Qwen2.5-7B.xlsx` | Same structure for 7B |
| `results_GPT4o-mini.xlsx` | GPT baseline + comparison vs best Qwen per size |

Export a single workbook:

```bash
python "Finetuning experiment/scripts/export_finetuning_report.py" --model 7B
```

Custom paths:

```bash
python "Finetuning experiment/scripts/export_finetuning_report.py" \
  --results-dir "Finetuning experiment/results" \
  --output-dir "Finetuning experiment/report"
```

## Requirements

Uses `pandas` and `openpyxl` from the repo root [`requirements.txt`](../../requirements.txt).

## Registered runs

Runs are defined in [`run_registry.json`](run_registry.json). The export script **fails** if any registered run is missing `metrics_summary.json`.

**3B** (`Qwen2.5-3B/`):

- `qwen_base` — base model
- `qwen_lora` — LoRA, 1 epoch, few-shot
- `qwen_lora_no_few_shots_2_epochs` — LoRA, 2 epochs, no few-shot
- `qwen_lora_no_few_shots_3_epochs` — LoRA, 3 epochs, no few-shot

**7B** (`Qwen2.5-7B/`): same folder names.

**GPT**: `gpt_base/`

To add a run, edit `run_registry.json` and re-run the export script.

## Workbook sheets

### Qwen workbooks (3B / 7B)

1. **main_results** — language pairs as rows; BLEU, chrF, term accuracy, total terms; absolute deltas vs GPT and vs base
2. **base_vs_lora** — best LoRA config per language vs base
3. **epoch_ablation** — `lora_2ep_nofs` vs `lora_3ep_nofs`
4. **experiment_config** — run metadata and file existence flags
5. **training_loss_meta** — final step/loss per run
6. **training_loss** — wide step × run loss table

### GPT workbook

1. **gpt_baseline** — GPT scores per language
2. **gpt_vs_best_qwen** — best 3B and 7B LoRA vs GPT per language
3. **experiment_config** — GPT run metadata

## Scripts

| File | Role |
|------|------|
| `run_registry.json` | Which result folders to include |
| `metrics_parser.py` | Load JSON metrics and training loss files |
| `sheet_builders.py` | Build DataFrames for each sheet |
| `export_finetuning_report.py` | CLI entry point |

## Notes

- Only `proper_term` mode is exported (all finetuning runs use this mode).
- Per-term ratio breakdowns are not exported (too large for Excel).
- Output is raw numeric data — no styling or conditional formatting.
- Legacy folder `qwen_lora_no_few_shots_two_epochs` is **not** used; canonical 3B 2-epoch path is `qwen_lora_no_few_shots_2_epochs`.
