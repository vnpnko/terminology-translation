# 02 · Expansion strategies

Compares the same term-expansion strategies as [`01_term_expansion`](../01_term_expansion/README.md) (original / GPT contextual expand / GPT domain-filtered clean, plus no_term, random_term, and external dictionary), broken out per language pair rather than aggregated. Produces the poster's `fig_exp23_expansion_strategies` figure.

This is a **thin wrapper**: no experiment-local data or results. It reads the same shared `data/` → `results/` pipeline as `01_term_expansion` from a different angle (per-language rather than macro-averaged).

## Reproduce

```bash
python src/analysis/generate_result_figures.py --only exp23
```

Generating script: [`src/analysis/figure_exp23.py`](../../src/analysis/figure_exp23.py) (`build_exp23_figure`), shared helpers in [`figure_common.py`](../../src/analysis/figure_common.py) and [`metrics_loader.py`](../../src/analysis/metrics_loader.py).

## Inputs

Same `results/dev_v1/{original,expand,cleaned,dictionary}/gpt/metrics_summary.json` files as `01_term_expansion` — see that experiment's table for the full source-data → results-path mapping.

## Output

[`poster/figures/fig_exp23_expansion_strategies.pdf`](../../poster/figures/fig_exp23_expansion_strategies.pdf)
