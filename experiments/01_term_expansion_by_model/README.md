# 01 · Term expansion strategies by model

Compares GPT-4o-mini translation quality across `no_term`, `random_term`, the three `proper_term` source variants (original / GPT-expanded / GPT-cleaned), and the external term dictionary. Produces the poster's `fig_exp1_term_expansion` figure.

This is a **thin wrapper**: there is no experiment-local data or results here. It is a specific comparison drawn from the shared, evolving `data/` → `results/` pipeline documented in the [root README](../../README.md#data), not a separate model run like [`04_baseline`](../04_baseline/README.md) or [`05_lora_finetuning`](../05_lora_finetuning/README.md).

## Reproduce

```bash
python src/analysis/generate_result_figures.py --only exp1
```

Generating script: [`src/analysis/figure_exp1.py`](../../src/analysis/figure_exp1.py) (`build_exp1_figure`), shared helpers in [`figure_common.py`](../../src/analysis/figure_common.py) and [`metrics_loader.py`](../../src/analysis/metrics_loader.py).

## Inputs

Reads `metrics_summary.json` under `results/dev_v1/` for the GPT-4o-mini baseline across:

| Variant | Source data | Results path |
| ------- | ------------ | ------------ |
| `original` | `data/raw/dev_v1_original/` | `results/dev_v1/original/gpt/` |
| `expand` | `data/interim/dev_v1_expand/` | `results/dev_v1/expand/gpt/` |
| `cleaned` | `data/interim/dev_v1_cleaned/` | `results/dev_v1/cleaned/gpt/` |
| external dictionary | `data/interim/dev_v1_dictionary/` | `results/dev_v1/dictionary/gpt/` |

Plus the `no_term` and `random_term` baseline modes from the same `original` results directory.

## Output

[`poster/figures/fig_exp1_term_expansion.pdf`](../../poster/figures/fig_exp1_term_expansion.pdf)
