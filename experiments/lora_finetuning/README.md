# LoRA fine-tuning

LoRA fine-tuning of Qwen2.5 (3B and 7B) on the `dev_v2` training set, compared against a Qwen base model and a GPT baseline, evaluated on `proper_term` mode only. Produces the poster's `fig_lora_epoch_ablation` and `fig_lora_best_models` figures, each owned by its matching sub-experiment (see below).

Five sub-experiments, each independently reproducible (own `scripts/`+`report/`), share this directory's `shared/results/`, `shared/data/`, and `shared/run_registry.json`:

| Sub-experiment | Contents |
| -------------- | -------- |
| [`epoch_ablation/`](epoch_ablation/README.md) | 1 / 2 / 3 LoRA epochs, one sheet per model size. |
| [`best_models/`](best_models/README.md) | Best LoRA config vs. GPT. |
| [`base_vs_lora/`](base_vs_lora/README.md) | `qwen_base_few_shot` vs. `qwen_lora_zero_shot` (1 epoch). |
| [`few_shot_ablation/`](few_shot_ablation/README.md) | `zero_shot` vs. `few_shot` prompting for GPT, Qwen base, and Qwen LoRA. |
| [`leakage_check/`](leakage_check/README.md) | Data-leakage honesty check: does LoRA training on `dev_v2` leak into the `dev_v1` test set? |

## Layout

| Path | Contents |
|------|----------|
| [`../../shared/notebooks/gpt.ipynb`](../../shared/notebooks/gpt.ipynb) | GPT baseline run on the finetuning test set |
| [`../../shared/notebooks/qwen_base.ipynb`](../../shared/notebooks/qwen_base.ipynb) | Qwen2.5 (3B/7B) base model, no fine-tuning |
| [`../../shared/notebooks/qwen_finetuned.ipynb`](../../shared/notebooks/qwen_finetuned.ipynb) | LoRA fine-tuning + inference for Qwen2.5 (3B/7B), 1/2/3 epochs |
| `shared/data/test/` | Held-out `dev_v1` test set per language pair |
| `shared/data/training/` | `dev_v2` training set per language pair |
| `shared/data/dev_v2_deduped/` | `dev_v2` with `dev_v1`-overlapping lines removed (see `shared/scripts/remove_dev_v2_overlap.py`); upstream of `shared/data/training/` |
| `shared/results/` | Per-model, per-run predictions and `metrics_summary.json` (see run names below) — shared by all 5 sub-experiments |
| `shared/run_registry.json` | Run registry (folder names, `use_few_shot`, epoch counts) — shared by all 5 sub-experiments |
| `shared/scripts/` | Shared data-prep script and `compare_common.py` (see "Scripts" below) — each sub-experiment has its own `scripts/` for its comparison script (and, for `epoch_ablation`/`best_models`, its figure script), importing from this `shared/scripts/compare_common.py` |
| `epoch_ablation/`, `best_models/`, `base_vs_lora/`, `few_shot_ablation/`, `leakage_check/` | Sub-experiments (see table above) |

## Running the notebooks

The three model-running notebooks live in the repo-root [`shared/notebooks/`](../../shared/notebooks/) directory (run manually, e.g. on LRZ AI Systems — not driven by any script). Each notebook writes predictions and `metrics_summary.json` under `shared/results/<model>/<run_name>/`. Run order: `qwen_base.ipynb` and `gpt.ipynb` first (baselines), then `qwen_finetuned.ipynb` for the LoRA runs.

### Registered runs

Runs are defined in [`shared/run_registry.json`](shared/run_registry.json). Run ids use the `zero_shot`/`few_shot` vocabulary (see "Naming standard" below), which does not match the on-disk result folder names — see the mapping below.

**3B** (`Qwen2.5-3B/`):

- `base_few_shot` (folder `qwen_base`) — base model, few-shot
- `lora_1_epoch_few_shot` (folder `qwen_lora`) — LoRA, 1 epoch, few-shot
- `lora_1_epoch_zero_shot` (folder `qwen_lora_no_few_shots`) — LoRA, 1 epoch, zero-shot
- `lora_2_epoch_zero_shot` (folder `qwen_lora_no_few_shots_2_epochs`) — LoRA, 2 epochs, zero-shot
- `lora_3_epoch_zero_shot` (folder `qwen_lora_no_few_shots_3_epochs`) — LoRA, 3 epochs, zero-shot

**7B** (`Qwen2.5-7B/`): same folder names, same run ids.

**GPT**: `gpt_base/` — run id `gpt_base` (always few-shot).

### Naming standard

This experiment's scripts, `run_registry.json`, and generated workbooks use `zero_shot`/`few_shot` as the one term for few-shot-prompting status, and canonical name pairs for the two models compared throughout: `gpt` (machine key: folder/dict keys, `model` column values) / `GPT` (prose and chart-label display), and `qwen_3b`/`qwen_7b` (machine key) / `Qwen 3B`/`Qwen 7B` (prose and chart-label display). Workbook sheet titles are the one exception — they use the full `GPT-4o-mini`/`Qwen2.5-3B`/`Qwen2.5-7B` form, mirroring the literal `run_registry.json` `model_dir` folder names rather than the short display form. Both `base` runs use `use_few_shot: true` in `run_registry.json` — the base runs are in fact few-shot (3 examples per language, per `report/README.md`), and the run id is `base_few_shot` accordingly. The shared `shared/results/dev_v1/original/{zero_shot,few_shot}/` folders (also read by the `term_expansion/` experiments) use this same vocabulary.

## Output

This experiment produces five figures, each living with the sub-experiment it belongs to:

- [`epoch_ablation/figures/fig_lora_epoch_ablation.pdf`](epoch_ablation/figures/fig_lora_epoch_ablation.pdf) — BLEU/term accuracy vs. LoRA epoch count, both model sizes. See [`epoch_ablation/README.md`](epoch_ablation/README.md).
- [`best_models/figures/fig_lora_best_models.pdf`](best_models/figures/fig_lora_best_models.pdf) — best LoRA config vs. GPT, by language pair. See [`best_models/README.md`](best_models/README.md).
- [`leakage_check/figures/fig_lora_leakage_check.pdf`](leakage_check/figures/fig_lora_leakage_check.pdf) — BLEU/term accuracy, grouped by model, overlap-data vs no-overlap-data bars. See [`leakage_check/README.md`](leakage_check/README.md).
- [`few_shot_ablation/figures/fig_lora_few_shot_ablation.pdf`](few_shot_ablation/figures/fig_lora_few_shot_ablation.pdf) — GPT only, all 5 metrics grouped with zero_shot/few_shot bars. See [`few_shot_ablation/README.md`](few_shot_ablation/README.md).
- [`few_shot_ablation/figures/fig_lora_few_shot_ablation_qwen.pdf`](few_shot_ablation/figures/fig_lora_few_shot_ablation_qwen.pdf) — Qwen only, 2×2 grid (3B/7B × base/LoRA), each panel grouped by zero_shot/few_shot with 5 metric bars. See [`few_shot_ablation/README.md`](few_shot_ablation/README.md).

The first two are copied to [`poster/figures/`](../../poster/figures/) for the poster; `leakage_check`'s and both `few_shot_ablation` figures are not currently included there.

## Scripts

| File | Role |
|------|------|
| `shared/scripts/remove_dev_v2_overlap.py` | Removes `dev_v2` lines whose English source also appears in `dev_v1/dev_v1_original` (writes `shared/data/dev_v2_deduped/`), upstream of the shared `shared/data/training/` used by all 5 sub-experiments' LoRA runs |
| `shared/scripts/compare_common.py` | Shared helpers (`load_registry`, `load_metrics_summary`, `extract_metric`, `extract_row`, `write_group_header`, the `METRICS` spec, and named run-id constants) imported by all 5 sub-experiments' `compare_*_to_excel.py` scripts — not a script itself, just deduplicated plumbing |

See each sub-experiment's README for its own comparison/filter script.

### Requirements

Uses `openpyxl` from the repo root [`requirements.txt`](../../requirements.txt).

### Notes

- Only `proper_term` mode is exported (all finetuning runs use this mode).
- Per-term ratio breakdowns are not exported (too large for Excel).
- Output is raw numeric data — no styling or conditional formatting beyond the Good/Bad/Neutral cell coloring (see `report/README.md`'s coloring-conventions section).
- Folder `qwen_lora_no_few_shots_two_epochs` is **not** used; the canonical 3B 2-epoch path is `qwen_lora_no_few_shots_2_epochs`.
