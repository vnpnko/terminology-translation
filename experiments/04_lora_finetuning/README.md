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

Runs are defined in [`scripts/run_registry.json`](scripts/run_registry.json). Run ids use the `zero_shot`/`few_shot` vocabulary (see "Naming standard" below), which does not match the on-disk result folder names — see the mapping below.

**3B** (`Qwen2.5-3B/`):

- `base_few_shot` (folder `qwen_base`) — base model, few-shot
- `lora_1_epoch_few_shot` (folder `qwen_lora`) — LoRA, 1 epoch, few-shot
- `lora_1_epoch_zero_shot` (folder `qwen_lora_no_few_shots`) — LoRA, 1 epoch, zero-shot
- `lora_2_epoch_zero_shot` (folder `qwen_lora_no_few_shots_2_epochs`) — LoRA, 2 epochs, zero-shot
- `lora_3_epoch_zero_shot` (folder `qwen_lora_no_few_shots_3_epochs`) — LoRA, 3 epochs, zero-shot

**7B** (`Qwen2.5-7B/`): same folder names, same run ids.

**GPT**: `gpt_base/` — run id `gpt_base` (always few-shot).

### Naming standard

This experiment's scripts, `run_registry.json`, and generated workbooks use `zero_shot`/`few_shot` as the one term for few-shot-prompting status, and canonical name pairs for the two models compared throughout: `gpt` (machine key: folder/dict keys, `model` column values) / `GPT-4o-mini` (display: sheet titles, group headers, prose), and `qwen_3b`/`qwen_7b` (machine key) / `Qwen2.5-3B`/`Qwen2.5-7B` (display). Both `base` runs use `use_few_shot: true` in `run_registry.json` — the base runs are in fact few-shot (3 examples per language, per `report/README.md`), and the run id is `base_few_shot` accordingly. The shared `results/dev_v1/original/{zero_shot,few_shot}/` folders (also read by `01`–`03`) use this same vocabulary.

## Report tables

`report/` holds one comparison workbook per axis, each with the same 5 metric columns (`BLEU`, `chrF`, `Term Acc (%)`, `Cons Macro Avg`, `Cons Weighted Avg`) evaluated on `proper_term` mode, one row per `lang_pair`:

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/leakage_honesty_check.xlsx` | Data-leakage honesty check (§3.4.2), single sheet `overlap_vs_no_overlap_data` with `overlap_data`/`no_overlap_data` column groups (named for the split criterion — ≥50% token containment with the training set — not a "bad"/"good" judgment) and 3 stacked 3-row model blocks in this order: `qwen_base` (untrained, control), `gpt` (closed model, never exposed to `dev_v2` training data, control), `qwen_lora` (trained, the model under test) — controls first so the overlap-vs-no-overlap gap visibly grows with training exposure; each cell colored green (higher/tied-highest) or yellow (lower) against its overlap_data/no_overlap_data counterpart, no red | `python experiments/04_lora_finetuning/scripts/compare_leakage_honesty_check_to_excel.py` — LoRA run selectable via `--lora-run` (default `lora_2_epoch_zero_shot`) |
| `report/base_few_shot_vs_lora_zero_shot_1_epoch.xlsx` | `qwen_base_few_shot` vs `qwen_lora_zero_shot` (1 epoch), for 3B and 7B; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/04_lora_finetuning/scripts/compare_base_vs_lora_to_excel.py` |
| `report/best_models.xlsx` | Best LoRA config (`qwen_lora_7b_zero_shot_2_epochs`, see note below) vs GPT-4o-mini; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/04_lora_finetuning/scripts/compare_best_models_to_excel.py` — LoRA run selectable via `--lora-run` (default `lora_2_epoch_zero_shot`) |
| `report/epoch_ablation.xlsx` | 1 / 2 / 3 LoRA epochs (zero-shot), one sheet per model size; each cell colored red/yellow/green by ranking that (language, metric) value across the 3 epochs (worst/mid/best) | `python experiments/04_lora_finetuning/scripts/compare_epochs_to_excel.py` |
| `report/zero_shot_vs_few_shot_ablation.xlsx` | `zero_shot` vs `few_shot`, one sheet per model (GPT-4o-mini, Qwen 3B, Qwen 7B); Qwen sheets stack `qwen_base` and `qwen_lora` rows; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/04_lora_finetuning/scripts/compare_few_shots_to_excel.py` — GPT/`qwen_base` rows split across two results trees (see script docstring); `qwen_lora` rows are fully local, driven by `run_registry.json` |

All 5 files are script-generated — no hand-assembled workbooks are in `report/`.

**Note:** `compare_few_shots_to_excel.py` fills in `Qwen2.5-7B`'s `qwen_base` `zero_shot` cells from `results/dev_v1/original/zero_shot/qwen_7b/`.

**Note:** `compare_best_models_to_excel.py` defaults to `lora_2_epoch_zero_shot` (override with `--lora-run`). This follows the evidence in `epoch_ablation.xlsx`: going from 2 to 3 epochs drops training loss much further (~0.05–0.07 vs ~0.14–0.17) while held-out BLEU/chrF plateau or slightly regress and only Term Accuracy keeps rising — an overfitting signature, so 2 epochs is treated as the better generalizing checkpoint.

**Note:** `leakage_honesty_check.xlsx` is fully script-generated (`compare_leakage_honesty_check_to_excel.py`), with column groups named `overlap_data`/`no_overlap_data` for the actual split criterion (≥50% token containment with the training set), not a "bad"/"good" value judgment. The `test_cleaned_by_sentences/data_good` split it reads for `qwen_base` is a real, verified good-filtered subset.

**Model names in `leakage_honesty_check.xlsx`:** the workbook's `model` column uses short labels; here's what each one is concretely: `gpt` = GPT-4o-mini; `qwen_base` = Qwen2.5-7B, base model, few-shot (`run_registry.json`'s `base_few_shot`, folder `qwen_base` — the folder name itself doesn't indicate few-shot, but the run is); `qwen_lora` = Qwen2.5-7B, LoRA fine-tuned, 2 epochs, zero-shot (`run_registry.json`'s `lora_2_epoch_zero_shot`, folder `qwen_lora_no_few_shots_2_epochs`, the script's `--lora-run` default).

### Requirements

Uses `openpyxl` from the repo root [`requirements.txt`](../../requirements.txt).

### Scripts

| File | Role |
|------|------|
| `scripts/compare_leakage_honesty_check_to_excel.py` | Builds `report/leakage_honesty_check.xlsx` (single sheet, see "Report tables" above) from `metrics_summary.json` under `.../test_cleaned_by_sentences/{data_bad,data_good}/`; LoRA run overridable via `--lora-run` |
| `scripts/compare_epochs_to_excel.py` | Builds `report/epoch_ablation.xlsx` (1/2/3 zero-shot LoRA epochs, one sheet per model size, rank-colored red/yellow/green) from `metrics_summary.json`, run selection driven by `run_registry.json` |
| `scripts/compare_few_shots_to_excel.py` | Builds `report/zero_shot_vs_few_shot_ablation.xlsx` (zero_shot vs few_shot for GPT, Qwen base, Qwen LoRA) from `metrics_summary.json` in both `results/` (root) and `experiments/04_lora_finetuning/results/` |
| `scripts/compare_base_vs_lora_to_excel.py` | Builds `report/base_few_shot_vs_lora_zero_shot_1_epoch.xlsx` (qwen_base_few_shot vs qwen_lora_zero_shot, one sheet, 3B/7B blocks) from `metrics_summary.json`, run selection driven by `run_registry.json` |
| `scripts/compare_best_models_to_excel.py` | Builds `report/best_models.xlsx` (best Qwen 7B LoRA config, default 2 epochs zero-shot, vs GPT-4o-mini) from `metrics_summary.json`; LoRA run overridable via `--lora-run` |
| `scripts/filter_test_sentence_overlap.py` | Filters test sentences overlapping with training data |
| `scripts/filter_test_term_overlap.py` | Filters test terms overlapping with training data |
| `scripts/figure_lora_finetuning.py` | Builds the poster's `fig_lora_finetuning` figure (`build_lora_finetuning_figure`; run via `python src/analysis/generate_result_figures.py --only lora_finetuning`) |
| `scripts/remove_dev_v2_overlap.py` | Removes `dev_v2` lines whose English source also appears in `dev_v1/dev_v1_original` (writes `data/dev_v2_deduped/`); feeds the leakage honesty-check comparisons |
| `scripts/run_registry.json` | Run registry (folder names, `use_few_shot`, epoch counts) driving `compare_epochs_to_excel.py`, `compare_base_vs_lora_to_excel.py`, `compare_best_models_to_excel.py`, and `compare_leakage_honesty_check_to_excel.py`; also carries unused `workbook` fields (see next row) |
| `scripts/metrics_parser.py`, `scripts/sheet_builders.py`, `scripts/export_finetuning_report.py` | Export pipeline for a `results_Qwen2.5-{3B,7B}.xlsx` / `results_GPT4o-mini.xlsx` workbook format; produces nothing currently used in `report/` (see [`report/ISSUES.md`](../../report/ISSUES.md)) |

### Notes

- Only `proper_term` mode is exported (all finetuning runs use this mode).
- Per-term ratio breakdowns are not exported (too large for Excel).
- Output is raw numeric data — no styling or conditional formatting.
- Folder `qwen_lora_no_few_shots_two_epochs` is **not** used; the canonical 3B 2-epoch path is `qwen_lora_no_few_shots_2_epochs`.
