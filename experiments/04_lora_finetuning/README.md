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

Runs are defined in [`scripts/run_registry.json`](scripts/run_registry.json). Run ids use the standard `zero_shot`/`few_shot` vocabulary (see "Naming standard" below); the underlying result folders keep their original names (only the registry's `run_id`s and `label`s were renamed, not the on-disk directories).

**3B** (`Qwen2.5-3B/`):

- `base_few_shot` (folder `qwen_base`) — base model, few-shot
- `lora_1_epoch_few_shot` (folder `qwen_lora`) — LoRA, 1 epoch, few-shot
- `lora_1_epoch_zero_shot` (folder `qwen_lora_no_few_shots`) — LoRA, 1 epoch, zero-shot
- `lora_2_epoch_zero_shot` (folder `qwen_lora_no_few_shots_2_epochs`) — LoRA, 2 epochs, zero-shot
- `lora_3_epoch_zero_shot` (folder `qwen_lora_no_few_shots_3_epochs`) — LoRA, 3 epochs, zero-shot

**7B** (`Qwen2.5-7B/`): same folder names, same run ids.

**GPT**: `gpt_base/` — run id `gpt_base` (always few-shot).

### Naming standard

This experiment's scripts, `run_registry.json`, and generated workbooks use `zero_shot`/`few_shot` (never `no_shots`/`few_shots`/`with_few_shots`/`no-few-shots`/`fs`/`nofs`) as the one standard term for few-shot-prompting status, and canonical name pairs for the two models compared throughout: `gpt` (machine key: folder/dict keys, `model` column values) / `GPT-4o-mini` (display: sheet titles, group headers, prose), and `qwen_3b`/`qwen_7b` (machine key) / `Qwen2.5-3B`/`Qwen2.5-7B` (display). This also fixed `run_registry.json`'s `use_few_shot: false` flag on both `base` runs — it was wrong (the base runs are actually few-shot, per `report/README.md` §3.4.1 point 3) — the flag is now `true` and the run id renamed `base_few_shot` to make it self-documenting. The shared `results/dev_v1/original/{no-few-shots,with-few-shots}/` folders (read by `01`-`03` too) were renamed to `results/dev_v1/original/{zero_shot,few_shot}/` to match.

## Report tables

`report/` holds one comparison workbook per axis, each with the same 5 metric columns (`BLEU`, `chrF`, `Term Acc (%)`, `Cons Macro Avg`, `Cons Weighted Avg`) evaluated on `proper_term` mode, one row per `lang_pair`:

| File | Compares | Regenerate |
| ---- | -------- | ---------- |
| `report/leakage_honesty_check.xlsx` | Data-leakage honesty check (§3.4.2), single sheet `overlap_vs_no_overlap_data` with `overlap_data`/`no_overlap_data` column groups (named for the split criterion — ≥50% token containment with the training set — not a "bad"/"good" judgment) and 3 stacked 3-row model blocks in this order: `qwen_base` (untrained, control), `gpt` (closed model, never exposed to `dev_v2` training data, control), `qwen_lora` (trained, the model under test) — controls first so the overlap-vs-no-overlap gap visibly grows with training exposure; each cell colored green (higher/tied-highest) or yellow (lower) against its overlap_data/no_overlap_data counterpart, no red | `python experiments/04_lora_finetuning/scripts/compare_leakage_honesty_check_to_excel.py` — LoRA run selectable via `--lora-run` (default `lora_2_epoch_zero_shot`) |
| `report/base_few_shot_vs_lora_zero_shot_1_epoch.xlsx` | `qwen_base_few_shot` vs `qwen_lora_zero_shot` (1 epoch), for 3B and 7B; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/04_lora_finetuning/scripts/compare_base_vs_lora_to_excel.py` |
| `report/best_models.xlsx` | Best LoRA config (`qwen_lora_7b_zero_shot_2_epochs`, see note below) vs GPT-4o-mini; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/04_lora_finetuning/scripts/compare_best_models_to_excel.py` — LoRA run selectable via `--lora-run` (default `lora_2_epoch_zero_shot`) |
| `report/epoch_ablation.xlsx` | 1 / 2 / 3 LoRA epochs (zero-shot), one sheet per model size; each cell colored red/yellow/green by ranking that (language, metric) value across the 3 epochs (worst/mid/best) | `python experiments/04_lora_finetuning/scripts/compare_epochs_to_excel.py` |
| `report/zero_shot_vs_few_shot_ablation.xlsx` | `zero_shot` vs `few_shot`, one sheet per model (GPT-4o-mini, Qwen 3B, Qwen 7B); Qwen sheets stack `qwen_base` and `qwen_lora` rows; each cell colored green (higher/tied-highest) or yellow (lower), no red | `python experiments/04_lora_finetuning/scripts/compare_few_shots_to_excel.py` — GPT/`qwen_base` rows split across two results trees (see script docstring); `qwen_lora` rows are fully local, driven by `run_registry.json` |

All 5 files are now script-generated — no hand-assembled workbooks remain in `report/`.

**Note:** `compare_few_shots_to_excel.py` fixed two data bugs found in the previous hand-assembled `few_shots_ablation.xlsx` while verifying against it — a 3rd-row `enru` label mistakenly entered as `ende` in both the `qwen_base` and `qwen_lora` blocks (3B sheet), and a genuine `enes`/`enru` term-accuracy value swap in the `qwen_base` block (3B sheet). It also fills in `Qwen2.5-7B`'s `qwen_base` `zero_shot` cells, which the old file left blank even though the source data (`results/dev_v1/original/zero_shot/qwen_7b/`) exists.

**Note:** `best_models.xlsx` previously cited the 3-epoch LoRA run as "best". Comparing `epoch_ablation.xlsx`'s 2-vs-3-epoch numbers against each run's `training_loss.txt` showed training loss dropping much further at 3 epochs (~0.05–0.07 vs ~0.14–0.17 at 2 epochs) while held-out BLEU/chrF plateau or slightly regress and only Term Accuracy keeps rising — an overfitting signature. `compare_best_models_to_excel.py` now defaults to `lora_2_epoch_zero_shot` (override with `--lora-run`), and also fills GPT's previously-blank consistency columns.

**Note:** `good vs bad data.xlsx` was renamed and redesigned into `leakage_honesty_check.xlsx`. It went through several redesign rounds: first split into 2 sheets (`train_test_overlap`, `data_difficulty`), which duplicated the `qwen_lora` block in both; then consolidated back into a single sheet with all 3 models stacked once, controls-then-trained-model order; then the column groups (originally `bad_data`/`good_data`) were renamed to `overlap_data`/`no_overlap_data` (sheet: `overlap_vs_no_overlap_data`) to name the actual split criterion instead of a "bad"/"good" value judgment. It's fully script-generated (`fill_good_vs_bad_gpt.py`, which only covered 2 of the old 5 blocks, has been removed). The `test_cleaned_by_sentences/data_good` split for `qwen_base` — previously flagged as buggy/missing in `report/ISSUES.md` — was verified to exist and match the old file's numbers exactly, so that concern is resolved.

**Model names in `leakage_honesty_check.xlsx`:** the workbook's `model` column uses short labels; here's what each one is concretely: `gpt` = GPT-4o-mini; `qwen_base` = Qwen2.5-7B, base model, few-shot (`run_registry.json`'s `base_few_shot`, folder `qwen_base` — the folder name itself doesn't indicate few-shot, but the run is; this used to be silently wrong in the registry, see "Naming standard" above); `qwen_lora` = Qwen2.5-7B, LoRA fine-tuned, 2 epochs, zero-shot (`run_registry.json`'s `lora_2_epoch_zero_shot`, folder `qwen_lora_no_few_shots_2_epochs`, the script's `--lora-run` default).

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
| `scripts/run_registry.json`, `scripts/metrics_parser.py`, `scripts/sheet_builders.py`, `scripts/export_finetuning_report.py` | Legacy export pipeline for a `results_Qwen2.5-{3B,7B}.xlsx` / `results_GPT4o-mini.xlsx` workbook format that was superseded by the "Report tables" above and is no longer used to produce anything in `report/` (see [`report/ISSUES.md`](../../report/ISSUES.md)) |

### Notes

- Only `proper_term` mode is exported (all finetuning runs use this mode).
- Per-term ratio breakdowns are not exported (too large for Excel).
- Output is raw numeric data — no styling or conditional formatting.
- Legacy folder `qwen_lora_no_few_shots_two_epochs` is **not** used; canonical 3B 2-epoch path is `qwen_lora_no_few_shots_2_epochs` (result folder names are unchanged by the `zero_shot`/`few_shot` naming standardization — only `run_registry.json` run ids, script identifiers, and workbook labels were renamed).
