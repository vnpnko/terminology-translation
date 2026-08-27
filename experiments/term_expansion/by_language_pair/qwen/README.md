# Term expansion by language pair (Qwen2.5-3B vs. 7B)

Compares the two Qwen2.5 model sizes directly, broken out by language pair (`ende` / `enru` / `enes`) and term-list variant. Reuses the parent [`term_expansion/`](../../README.md)'s shared `shared/data/` → `shared/results/` pipeline and its `../shared/scripts/` compare scripts — see that README for the Inputs table. For the equivalent single-model (GPT-4o-mini) comparison, see the sibling [`gpt/`](../gpt/README.md).

## Reproduce

```bash
python shared/lib/analysis/generate_result_figures.py --only by_language_pair_qwen
```

Generating script: [`scripts/figure_qwen_size_comparison.py`](scripts/figure_qwen_size_comparison.py) (`build_qwen_size_stacked_figure`), which reuses data-loading and constants from [`../shared/scripts/by_language_pair_common.py`](../shared/scripts/by_language_pair_common.py), plus shared plotting helpers in [`shared/lib/analysis/grouped_bar_figure_common.py`](../../../../shared/lib/analysis/grouped_bar_figure_common.py) and [`shared/lib/analysis/metrics_loader.py`](../../../../shared/lib/analysis/metrics_loader.py).

## Output

[`figures/fig_term_expansion_qwen_size_stacked.pdf`](figures/fig_term_expansion_qwen_size_stacked.pdf) — grouped by language pair, one stacked bar per term-list variant: the solid segment is Qwen2.5-3B's score, the lighter segment on top is the increment needed to reach Qwen2.5-7B's score (valid since 7B's score is at or above 3B's in every cell but one, a near-tie explained in the report). Used in the report's Appendix.

## Report tables

The `language_comparison.xlsx` / `proper_term_across_languages.xlsx` tables (all three models, not Qwen-specific) live at the parent [`by_language_pair/report/`](../report/) — see the [`by_language_pair/README.md`](../README.md) Report tables section for regeneration commands.
