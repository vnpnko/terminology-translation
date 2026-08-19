# LoRA fine-tuning

LoRA fine-tuning of Qwen2.5 (3B and 7B) on the `dev_v2` training set, compared against a Qwen base model and a GPT-4o-mini baseline, evaluated on `proper_term` mode only. Produces the poster's `fig_lora_finetuning` figure (epoch ablation vs. GPT-4o-mini).

Five sub-experiments, each independently reproducible (own `scripts/`+`report/`), share this directory's `results/`, `data/`, and `run_registry.json`:

| Sub-experiment | Contents |
| -------------- | -------- |
| [`epoch_ablation/`](epoch_ablation/README.md) | 1 / 2 / 3 LoRA epochs, one sheet per model size. |
| [`best_models/`](best_models/README.md) | Best LoRA config vs. GPT-4o-mini. |
| [`base_vs_lora/`](base_vs_lora/README.md) | `qwen_base_few_shot` vs. `qwen_lora_zero_shot` (1 epoch). |
| [`few_shot_ablation/`](few_shot_ablation/README.md) | `zero_shot` vs. `few_shot` prompting for GPT, Qwen base, and Qwen LoRA. |
| [`leakage_check/`](leakage_check/README.md) | Data-leakage honesty check: does LoRA training on `dev_v2` leak into the `dev_v1` test set? |

## Layout

| Path | Contents |
|------|----------|
| [`../../notebooks/gpt.ipynb`](../../notebooks/gpt.ipynb) | GPT-4o-mini baseline run on the finetuning test set |
| [`../../notebooks/qwen_base.ipynb`](../../notebooks/qwen_base.ipynb) | Qwen2.5 (3B/7B) base model, no fine-tuning |
| [`../../notebooks/qwen_finetuned.ipynb`](../../notebooks/qwen_finetuned.ipynb) | LoRA fine-tuning + inference for Qwen2.5 (3B/7B), 1/2/3 epochs |
| `data/test/` | Held-out `dev_v1` test set per language pair |
| `data/training/` | `dev_v2` training set per language pair |
| `data/dev_v2_deduped/` | `dev_v2` with `dev_v1`-overlapping lines removed (see `scripts/remove_dev_v2_overlap.py`); upstream of `data/training/` |
| `results/` | Per-model, per-run predictions and `metrics_summary.json` (see run names below) — shared by all 5 sub-experiments |
| `run_registry.json` | Run registry (folder names, `use_few_shot`, epoch counts) — shared by all 5 sub-experiments |
| `figures/` | `fig_lora_finetuning`, this experiment's figure (see "Output" below) |
| `scripts/` | Shared data-prep script and the figure-generating script (see "Scripts" below) — each sub-experiment has its own `scripts/` for its comparison script |
| `epoch_ablation/`, `best_models/`, `base_vs_lora/`, `few_shot_ablation/`, `leakage_check/` | Sub-experiments (see table above) |

## Running the notebooks

The three model-running notebooks live in the repo-root [`notebooks/`](../../notebooks/) directory (run manually, e.g. on LRZ AI Systems — not driven by any script). Each notebook writes predictions and `metrics_summary.json` under `results/<model>/<run_name>/`. Run order: `qwen_base.ipynb` and `gpt.ipynb` first (baselines), then `qwen_finetuned.ipynb` for the LoRA runs.

### Registered runs

Runs are defined in [`run_registry.json`](run_registry.json). Run ids use the `zero_shot`/`few_shot` vocabulary (see "Naming standard" below), which does not match the on-disk result folder names — see the mapping below.

**3B** (`Qwen2.5-3B/`):

- `base_few_shot` (folder `qwen_base`) — base model, few-shot
- `lora_1_epoch_few_shot` (folder `qwen_lora`) — LoRA, 1 epoch, few-shot
- `lora_1_epoch_zero_shot` (folder `qwen_lora_no_few_shots`) — LoRA, 1 epoch, zero-shot
- `lora_2_epoch_zero_shot` (folder `qwen_lora_no_few_shots_2_epochs`) — LoRA, 2 epochs, zero-shot
- `lora_3_epoch_zero_shot` (folder `qwen_lora_no_few_shots_3_epochs`) — LoRA, 3 epochs, zero-shot

**7B** (`Qwen2.5-7B/`): same folder names, same run ids.

**GPT**: `gpt_base/` — run id `gpt_base` (always few-shot).

### Naming standard

This experiment's scripts, `run_registry.json`, and generated workbooks use `zero_shot`/`few_shot` as the one term for few-shot-prompting status, and canonical name pairs for the two models compared throughout: `gpt` (machine key: folder/dict keys, `model` column values) / `GPT-4o-mini` (display: sheet titles, group headers, prose), and `qwen_3b`/`qwen_7b` (machine key) / `Qwen2.5-3B`/`Qwen2.5-7B` (display). Both `base` runs use `use_few_shot: true` in `run_registry.json` — the base runs are in fact few-shot (3 examples per language, per `report/README.md`), and the run id is `base_few_shot` accordingly. The shared `results/dev_v1/original/{zero_shot,few_shot}/` folders (also read by the `term_expansion/` experiments) use this same vocabulary.

## Output

[`figures/fig_lora_finetuning.pdf`](figures/fig_lora_finetuning.pdf) — this is the figure's home; it's copied to [`poster/figures/`](../../poster/figures/fig_lora_finetuning.pdf) for the poster. Draws on the same underlying data as `epoch_ablation/` (epoch line-chart panels) and `best_models/`-like data (final GPT-vs-best-LoRA comparison panels) — not code-coupled to either sub-experiment's script, since the figure reads `metrics_summary.json` independently, but conceptually spans both, which is why it stays at this parent level rather than in either sub-experiment.

## Scripts

| File | Role |
|------|------|
| `scripts/figure_lora_finetuning.py` | Builds the poster's `fig_lora_finetuning` figure (`build_lora_finetuning_figure`; run via `python src/analysis/generate_result_figures.py --only lora_finetuning`); imports `metrics_parser.py` for `metrics_summary.json` loading/aggregation helpers |
| `scripts/metrics_parser.py` | `metrics_summary.json` loading/aggregation helpers, used only by `figure_lora_finetuning.py` |
| `scripts/remove_dev_v2_overlap.py` | Removes `dev_v2` lines whose English source also appears in `dev_v1/dev_v1_original` (writes `data/dev_v2_deduped/`), upstream of the shared `data/training/` used by all 5 sub-experiments' LoRA runs |

See each sub-experiment's README for its own comparison/filter script.

### Requirements

Uses `openpyxl` from the repo root [`requirements.txt`](../../requirements.txt).

### Notes

- Only `proper_term` mode is exported (all finetuning runs use this mode).
- Per-term ratio breakdowns are not exported (too large for Excel).
- Output is raw numeric data — no styling or conditional formatting beyond the Good/Bad/Neutral cell coloring (see `report/README.md`'s coloring-conventions section).
- Folder `qwen_lora_no_few_shots_two_epochs` is **not** used; the canonical 3B 2-epoch path is `qwen_lora_no_few_shots_2_epochs`.
