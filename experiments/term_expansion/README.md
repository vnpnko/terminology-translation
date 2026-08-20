# Term expansion

Compares terminology-constrained translation across `no_term`, `random_term`, the three `proper_term` source variants (original / GPT-expanded / GPT-cleaned), and the external term dictionary, from two angles: aggregated by model, and broken out by language pair. Produces the poster's `fig_term_expansion_across_model` figure (aggregated by model) and the 3 `fig_term_expansion_across_languages_{gpt,qwen_3b,qwen_7b}` figures (by language pair, one per model).

This is a **thin wrapper**: there is no experiment-local data here, only this experiment's `scripts/`, `figures/`, and `report/` below. It is a specific comparison drawn from the shared, evolving `data/` → `results/` pipeline documented in the [root README](../../README.md#data), not a separate model run like [`lora_finetuning`](../lora_finetuning/README.md).

The externally-sourced dictionary term-list variant (`results/dev_v1/dictionary/`) doesn't have its own experiment folder — its data-prep scripts and standalone comparisons are folded in here.

For the `dev_v1`-vs-`dev_v2` dataset comparability check (a different axis — not about term-list strategy), see the top-level [`dataset_comparison/`](../dataset_comparison/README.md).

## Reproduce

```bash
python src/analysis/generate_result_figures.py --only model_comparison mode_comparison
```

Generating scripts: [`scripts/figure_by_model.py`](scripts/figure_by_model.py) (`build_by_model_figure`) and [`scripts/figure_by_language_pair.py`](scripts/figure_by_language_pair.py) (`build_by_language_pair_figures`, one figure per model), shared helpers in [`src/analysis/figure_common.py`](../../src/analysis/figure_common.py) and [`src/analysis/metrics_loader.py`](../../src/analysis/metrics_loader.py).

## Inputs

Reads `metrics_summary.json` under `results/dev_v1/` for all 3 models (GPT-4o-mini, Qwen 3B, Qwen 7B) across:

| Variant | Source data | Results path |
| ------- | ------------ | ------------ |
| `original` | `data/dev_v1/dev_v1_original/` | `results/dev_v1/original/few_shot/` |
| `expand` | `data/dev_v1/dev_v1_expand/` | `results/dev_v1/expand/` |
| `cleaned` | `data/dev_v1/dev_v1_cleaned/` | `results/dev_v1/cleaned/` |
| external dictionary | `data/dev_v1/dev_v1_dictionary/` | `results/dev_v2/` |

Plus the `no_term` and `random_term` baseline modes — only from `results/dev_v1/original/few_shot/`. These two modes don't depend on the term-list variant, so they were only ever run once and were pruned from `expand`/`cleaned`/`dictionary` as redundant; `metrics_summary.json` in those directories now has only a `proper_term` entry per language. `zero_shot` was similarly pruned to `proper_term` only — `few_shot` is the only `dev_v1/original` variant with all 3 modes, which is why both figures and both comparison scripts source `original` from there (not `zero_shot`).

`compare_by_model_and_language.py` (below) handles the pruned variants correctly: those sheets simply show `proper_term` rows only, with the mode-label merge computed from the rows actually present rather than assuming a fixed count per mode.

## Output

- [`figures/fig_term_expansion_across_model.pdf`](figures/fig_term_expansion_across_model.pdf) — aggregated by model (macro avg over language pairs), grouped by model, bars = strategy.
- `figures/fig_term_expansion_across_languages_{gpt,qwen_3b,qwen_7b}.pdf` — one figure per model, broken out by language pair, bars = strategy.

All 4 are copied to [`poster/figures/`](../../poster/figures/) for the poster.

## Report tables

Model-vs-model (GPT / Qwen 3B / Qwen 7B) and language-vs-language (`ende` / `enru` / `enes`) comparison workbooks, one sheet per dataset variant, generated in one run by [`scripts/compare_by_model_and_language.py`](scripts/compare_by_model_and_language.py):

| File | Regenerate |
| ---- | ---------- |
| `report/model_comparison.xlsx` (sheets: `dev_v1`, `dev_v2`; rows grouped by mode then language, columns per model) | `python experiments/term_expansion/scripts/compare_by_model_and_language.py` |
| `report/language_comparison.xlsx` (sheets: `dev_v1`, `dev_v2`; rows grouped by mode then model, columns per language) | (same command — writes both files) |

The `dev_v1_expand`/`dev_v1_cleaned`/`dev_v1_dictionary` sheets were dropped from these workbooks — since those variants only ever have `proper_term` data (no `no_term`/`random_term`), they were pure duplicates of the `proper_term` rows already in the workbooks below.

`proper_term`-only comparison across all 4 dev_v1 term-list variants (`original`/`expand`/`cleaned`/`dictionary`), generated in one run by [`scripts/compare_proper_term_by_model_and_language.py`](scripts/compare_proper_term_by_model_and_language.py). Both workbooks sit in one 12-row table (4 variants × 3 languages, or 4 variants × 3 models), ranked/colored together, same color scheme:

| File | Regenerate |
| ---- | ---------- |
| `report/proper_term_across_models.xlsx` (rows: variant × language, columns per model) | `python experiments/term_expansion/scripts/compare_proper_term_by_model_and_language.py` |
| `report/proper_term_across_languages.xlsx` (rows: variant × model, columns per language) | (same command — writes both files) |
