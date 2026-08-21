# Term expansion by model

Aggregated by model (GPT-4o-mini / Qwen 3B / Qwen 7B), macro avg over language pairs. Reuses the parent [`term_expansion/`](../README.md)'s shared `shared/data/` → `shared/results/` pipeline and its `../shared/scripts/` compare scripts — see that README for the Inputs table.

## Reproduce

```bash
python shared/lib/analysis/generate_result_figures.py --only model_comparison
```

Generating script: [`scripts/figure_by_model.py`](scripts/figure_by_model.py) (`build_by_model_figure`), shared helpers in [`shared/lib/analysis/grouped_bar_figure_common.py`](../../../shared/lib/analysis/grouped_bar_figure_common.py) and [`shared/lib/analysis/metrics_loader.py`](../../../shared/lib/analysis/metrics_loader.py).

## Output

[`figures/fig_term_expansion_across_model.pdf`](figures/fig_term_expansion_across_model.pdf) — grouped by model, bars = strategy. Copied to [`poster/figures/`](../../../poster/figures/) for the poster.

## Report tables

| File | Regenerate |
| ---- | ---------- |
| `report/model_comparison.xlsx` (sheets: `dev_v1`, `dev_v2`; rows grouped by mode then language, columns per model) | `python experiments/term_expansion/shared/scripts/compare_by_model_and_language.py` (also writes the sibling `by_language_pair/report/language_comparison.xlsx` in the same run) |
| `report/proper_term_across_models.xlsx` (rows: 4 term-list variants × language, columns per model) | `python experiments/term_expansion/shared/scripts/compare_by_model_and_language.py --mode proper_term` (also writes the sibling `by_language_pair/report/proper_term_across_languages.xlsx` in the same run) |

`model_comparison.xlsx` does not include `dev_v1_expand`/`dev_v1_cleaned`/`dev_v1_dictionary` sheets, since those variants only have `proper_term` data, which already appears in `proper_term_across_models.xlsx`.
