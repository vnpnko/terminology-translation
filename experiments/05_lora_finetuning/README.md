# 05 · LoRA fine-tuning

LoRA fine-tuning of Qwen2.5 (3B and 7B) on the `dev_v2` training set, compared against a Qwen base model and a GPT-4o-mini baseline, evaluated on `proper_term` mode only. Produces the poster's `fig_exp5_lora_finetuning` figure (epoch ablation vs. GPT-4o-mini).

## Layout

| Path | Contents |
|------|----------|
| `notebooks/gpt.ipynb` | GPT-4o-mini baseline run on the finetuning test set |
| `notebooks/qwen_base.ipynb` | Qwen2.5 (3B/7B) base model, no fine-tuning |
| `notebooks/qwen_finetuned.ipynb` | LoRA fine-tuning + inference for Qwen2.5 (3B/7B), 1/2/3 epochs |
| `data/test/` | Held-out `dev_v1` test set per language pair |
| `data/test_cleaned_by_sentences/`, `data/test_cleaned_by_terms/` | Test set filtered for overlap with the training set (see `scripts/filter_test_sentence_overlap.py` / `filter_test_term_overlap.py`) |
| `data/training/` | `dev_v2` training set per language pair |
| `results/` | Per-model, per-run predictions and `metrics_summary.json` (see run names below) |
| `report/` | Generated Excel workbooks (see below) |
| `scripts/` | Data filtering + Excel export scripts |

## Running the notebooks

Each notebook writes predictions and `metrics_summary.json` under `results/<model>/<run_name>/`. Run order: `qwen_base.ipynb` and `gpt.ipynb` first (baselines), then `qwen_finetuned.ipynb` for the LoRA runs.

## Excel export

Generate supervisor-ready Excel workbooks from `metrics_summary.json` and `training_loss.txt` files under `results/`.

From the repository root:

```bash
python experiments/05_lora_finetuning/scripts/export_finetuning_report.py
```

Outputs land in `report/`:

| File | Contents |
|------|----------|
| `results_Qwen2.5-3B.xlsx` | 3B runs: main results, base vs LoRA, epoch ablation, config, training loss |
| `results_Qwen2.5-7B.xlsx` | Same structure for 7B |
| `results_GPT4o-mini.xlsx` | GPT baseline + comparison vs best Qwen per size |

Export a single workbook:

```bash
python experiments/05_lora_finetuning/scripts/export_finetuning_report.py --model 7B
```

Custom paths:

```bash
python experiments/05_lora_finetuning/scripts/export_finetuning_report.py \
  --results-dir experiments/05_lora_finetuning/results \
  --output-dir experiments/05_lora_finetuning/report
```

### Requirements

Uses `pandas` and `openpyxl` from the repo root [`requirements.txt`](../../requirements.txt).

### Registered runs

Runs are defined in [`scripts/run_registry.json`](scripts/run_registry.json). The export script **fails** if any registered run is missing `metrics_summary.json`.

**3B** (`Qwen2.5-3B/`):

- `qwen_base` — base model
- `qwen_lora` — LoRA, 1 epoch, few-shot
- `qwen_lora_no_few_shots_2_epochs` — LoRA, 2 epochs, no few-shot
- `qwen_lora_no_few_shots_3_epochs` — LoRA, 3 epochs, no few-shot

**7B** (`Qwen2.5-7B/`): same folder names.

**GPT**: `gpt_base/`

To add a run, edit `scripts/run_registry.json` and re-run the export script.

### Workbook sheets

**Qwen workbooks (3B / 7B)**

1. **main_results** — language pairs as rows; BLEU, chrF, term accuracy, total terms; absolute deltas vs GPT and vs base
2. **base_vs_lora** — best LoRA config per language vs base
3. **epoch_ablation** — `lora_2ep_nofs` vs `lora_3ep_nofs`
4. **experiment_config** — run metadata and file existence flags
5. **training_loss_meta** — final step/loss per run
6. **training_loss** — wide step × run loss table

**GPT workbook**

1. **gpt_baseline** — GPT scores per language
2. **gpt_vs_best_qwen** — best 3B and 7B LoRA vs GPT per language
3. **experiment_config** — GPT run metadata

### Scripts

| File | Role |
|------|------|
| `scripts/run_registry.json` | Which result folders to include |
| `scripts/metrics_parser.py` | Load JSON metrics and training loss files |
| `scripts/sheet_builders.py` | Build DataFrames for each sheet |
| `scripts/export_finetuning_report.py` | CLI entry point for the Excel export |
| `scripts/filter_test_sentence_overlap.py` | Filters test sentences overlapping with training data |
| `scripts/filter_test_term_overlap.py` | Filters test terms overlapping with training data |
| `scripts/figure_exp5.py` | Builds the poster's `fig_exp5_lora_finetuning` figure (`build_exp5_figure`; run via `python src/analysis/generate_result_figures.py --only exp5`) |
| `scripts/remove_dev_v2_overlap.py` | Removes `dev_v2` lines whose English source also appears in `dev_v1/dev_v1_original` (writes `data/dev_v2_deduped/`); feeds the leakage honesty-check comparisons |

### Notes

- Only `proper_term` mode is exported (all finetuning runs use this mode).
- Per-term ratio breakdowns are not exported (too large for Excel).
- Output is raw numeric data — no styling or conditional formatting.
- Legacy folder `qwen_lora_no_few_shots_two_epochs` is **not** used; canonical 3B 2-epoch path is `qwen_lora_no_few_shots_2_epochs`.
