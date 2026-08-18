# 04 · LoRA fine-tuning

LoRA fine-tuning of Qwen2.5 (3B and 7B) on the `dev_v2` training set, compared against a Qwen base model and a GPT-4o-mini baseline, evaluated on `proper_term` mode only. Produces the poster's `fig_lora_finetuning` figure (epoch ablation vs. GPT-4o-mini).

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
| `report/` | Comparison Excel workbooks (see "Report tables" below) |
| `scripts/` | Data filtering + Excel report scripts |

## Running the notebooks

Each notebook writes predictions and `metrics_summary.json` under `results/<model>/<run_name>/`. Run order: `qwen_base.ipynb` and `gpt.ipynb` first (baselines), then `qwen_finetuned.ipynb` for the LoRA runs.

### Registered runs

Runs are defined in [`scripts/run_registry.json`](scripts/run_registry.json).

**3B** (`Qwen2.5-3B/`):

- `qwen_base` — base model
- `qwen_lora` — LoRA, 1 epoch, few-shot
- `qwen_lora_no_few_shots_2_epochs` — LoRA, 2 epochs, no few-shot
- `qwen_lora_no_few_shots_3_epochs` — LoRA, 3 epochs, no few-shot

- `qwen_lora_no_few_shots` — LoRA, 1 epoch, no few-shot

**7B** (`Qwen2.5-7B/`): same folder names.

**GPT**: `gpt_base/`

## Report tables

`report/` holds one comparison workbook per axis, each with the same 5 metric columns (`BLEU`, `chrF`, `Term Acc (%)`, `Cons Macro Avg`, `Cons Weighted Avg`) evaluated on `proper_term` mode, one row per `lang_pair`:

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/good vs bad data.xlsx` | GPT / Qwen-base / Qwen-LoRA(7B) on the leakage-honesty-check "good" vs "bad" test subsets (5 stacked blocks incl. GPT-vs-LoRA on each subset) | `python experiments/04_lora_finetuning/scripts/fill_good_vs_bad_gpt.py` — fills only the GPT and `qwen_base` blocks; the 3 `qwen_lora` blocks are hand-entered and must be updated manually |
| `report/base_few_shots_vs_lora_1_epoch.xlsx` | `qwen_base_with_few_shots` vs `qwen_lora_no_few_shots` (1 epoch), for 3B and 7B | no generating script — hand-assembled from `results/.../metrics_summary.json` |
| `report/best_models.xlsx` | Best LoRA config (`qwen_lora_7b_no_shots_3_epochs`) vs GPT-4o-mini few-shot | no generating script — hand-assembled |
| `report/epoch_ablation.xlsx` | 1 / 2 / 3 LoRA epochs (no few-shot), one sheet per model size; each cell colored red/yellow/green by ranking that (language, metric) value across the 3 epochs (worst/mid/best) | `python experiments/04_lora_finetuning/scripts/compare_epochs_to_excel.py` |
| `report/few_shots_ablation.xlsx` | `no_shots` vs `few_shots`, one sheet per model (GPT-4o-mini, Qwen 3B, Qwen 7B); Qwen sheets stack `qwen_base` and `qwen_lora` rows; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/04_lora_finetuning/scripts/compare_few_shots_to_excel.py` — GPT/`qwen_base` rows split across two results trees (see script docstring); `qwen_lora` rows are fully local, driven by `run_registry.json` |

**TODO:** the remaining 2 hand-assembled files (`base_few_shots_vs_lora_1_epoch.xlsx`, `best_models.xlsx`) still have no export script (unlike e.g. `experiments/03_dataset_comparison/scripts/compare_datasets_to_excel.py` or this experiment's own `compare_epochs_to_excel.py`/`compare_few_shots_to_excel.py`, which regenerate their workbooks from `metrics_summary.json` directly). Writing equivalent scripts for these would remove the manual-copy risk.

**Note:** `compare_few_shots_to_excel.py` fixed two data bugs found in the previous hand-assembled `few_shots_ablation.xlsx` while verifying against it — a 3rd-row `enru` label mistakenly entered as `ende` in both the `qwen_base` and `qwen_lora` blocks (3B sheet), and a genuine `enes`/`enru` term-accuracy value swap in the `qwen_base` block (3B sheet). It also fills in `Qwen2.5-7B`'s `qwen_base` `no_shots` cells, which the old file left blank even though the source data (`results/dev_v1/original/no-few-shots/qwen_7b/`) exists.

### Requirements

Uses `openpyxl` from the repo root [`requirements.txt`](../../requirements.txt).

### Scripts

| File | Role |
|------|------|
| `scripts/fill_good_vs_bad_gpt.py` | Fills the GPT and `qwen_base` blocks of `report/good vs bad data.xlsx` from `metrics_summary.json` |
| `scripts/compare_epochs_to_excel.py` | Builds `report/epoch_ablation.xlsx` (1/2/3 no-few-shot LoRA epochs, one sheet per model size, rank-colored red/yellow/green) from `metrics_summary.json`, run selection driven by `run_registry.json` |
| `scripts/compare_few_shots_to_excel.py` | Builds `report/few_shots_ablation.xlsx` (no_shots vs few_shots for GPT, Qwen base, Qwen LoRA) from `metrics_summary.json` in both `results/` (root) and `experiments/04_lora_finetuning/results/` |
| `scripts/filter_test_sentence_overlap.py` | Filters test sentences overlapping with training data |
| `scripts/filter_test_term_overlap.py` | Filters test terms overlapping with training data |
| `scripts/figure_lora_finetuning.py` | Builds the poster's `fig_lora_finetuning` figure (`build_lora_finetuning_figure`; run via `python src/analysis/generate_result_figures.py --only lora_finetuning`) |
| `scripts/remove_dev_v2_overlap.py` | Removes `dev_v2` lines whose English source also appears in `dev_v1/dev_v1_original` (writes `data/dev_v2_deduped/`); feeds the leakage honesty-check comparisons |
| `scripts/run_registry.json`, `scripts/metrics_parser.py`, `scripts/sheet_builders.py`, `scripts/export_finetuning_report.py` | Legacy export pipeline for a `results_Qwen2.5-{3B,7B}.xlsx` / `results_GPT4o-mini.xlsx` workbook format that was superseded by the "Report tables" above and is no longer used to produce anything in `report/` (see [`report/ISSUES.md`](../../report/ISSUES.md)) |

### Notes

- Only `proper_term` mode is exported (all finetuning runs use this mode).
- Per-term ratio breakdowns are not exported (too large for Excel).
- Output is raw numeric data — no styling or conditional formatting.
- Legacy folder `qwen_lora_no_few_shots_two_epochs` is **not** used; canonical 3B 2-epoch path is `qwen_lora_no_few_shots_2_epochs`.
