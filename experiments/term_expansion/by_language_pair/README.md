# Term expansion by language pair

Broken out by language pair (`ende` / `enru` / `enes`), one figure per model (GPT-4o-mini / Qwen 3B / Qwen 7B). Reuses the parent [`term_expansion/`](../README.md)'s shared `shared/data/` → `shared/results/` pipeline and its `../shared/scripts/` compare scripts — see that README for the Inputs table.

## Reproduce

```bash
python shared/lib/analysis/generate_result_figures.py --only mode_comparison
```

Generating script: [`scripts/figure_by_language_pair.py`](scripts/figure_by_language_pair.py) (`build_by_language_pair_figures`, one figure per model), shared helpers in [`shared/lib/analysis/grouped_bar_figure_common.py`](../../../shared/lib/analysis/grouped_bar_figure_common.py) and [`shared/lib/analysis/metrics_loader.py`](../../../shared/lib/analysis/metrics_loader.py).

## Output

`figures/fig_term_expansion_across_languages_{gpt,qwen_3b,qwen_7b}.pdf` — one figure per model, grouped by language pair, bars = strategy. Copied to [`poster/figures/`](../../../poster/figures/) for the poster.

## Report tables

| File | Regenerate |
| ---- | ---------- |
| `report/language_comparison.xlsx` (sheets: `dev_v1`, `dev_v2`; rows grouped by mode then model, columns per language) | `python experiments/term_expansion/shared/scripts/compare_by_model_and_language.py` (also writes the sibling `by_model/report/model_comparison.xlsx` in the same run) |
| `report/proper_term_across_languages.xlsx` (rows: 4 term-list variants × model, columns per language) | `python experiments/term_expansion/shared/scripts/compare_by_model_and_language.py --mode proper_term` (also writes the sibling `by_model/report/proper_term_across_models.xlsx` in the same run) |

`language_comparison.xlsx` does not include `dev_v1_expand`/`dev_v1_cleaned`/`dev_v1_dictionary` sheets, since those variants only have `proper_term` data, which already appears in `proper_term_across_languages.xlsx`.
