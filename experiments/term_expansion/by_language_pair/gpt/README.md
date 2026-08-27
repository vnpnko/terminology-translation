# Term expansion by language pair (GPT-4o-mini)

Broken out by language pair (`ende` / `enru` / `enes`), GPT-4o-mini only. Reuses the parent [`term_expansion/`](../../README.md)'s shared `shared/data/` → `shared/results/` pipeline and its `../shared/scripts/` compare scripts — see that README for the Inputs table. For the equivalent comparison across Qwen model sizes instead of language pair alone, see the sibling [`qwen/`](../qwen/README.md).

## Reproduce

```bash
python shared/lib/analysis/generate_result_figures.py --only by_language_pair_gpt
```

Generating script: [`scripts/figure_by_language_pair_gpt.py`](scripts/figure_by_language_pair_gpt.py) (`build_by_language_pair_gpt_figure`), which reuses data-loading and constants from [`../shared/scripts/by_language_pair_common.py`](../shared/scripts/by_language_pair_common.py), plus shared plotting helpers in [`shared/lib/analysis/grouped_bar_figure_common.py`](../../../../shared/lib/analysis/grouped_bar_figure_common.py) and [`shared/lib/analysis/metrics_loader.py`](../../../../shared/lib/analysis/metrics_loader.py).

## Output

[`figures/fig_term_expansion_across_languages_gpt.pdf`](figures/fig_term_expansion_across_languages_gpt.pdf) — grouped by language pair, bars = term-list variant. Used in the report's main body (Figure 1) and copied to [`poster/figures/`](../../../../poster/figures/) for the poster.

## Report tables

The `language_comparison.xlsx` / `proper_term_across_languages.xlsx` tables (all three models, not GPT-specific) live at the parent [`by_language_pair/report/`](../report/) — see the [`by_language_pair/README.md`](../README.md) Report tables section for regeneration commands.
