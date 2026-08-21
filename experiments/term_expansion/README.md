# Term expansion

Compares terminology-constrained translation across `no_term`, `random_term`, the three `proper_term` source variants (original / GPT-expanded / GPT-cleaned), and the external term dictionary, from two angles: aggregated by model, and broken out by language pair.

| Sub-experiment | Contents |
| -------------- | -------- |
| [`by_model/`](by_model/README.md) | Aggregated by model (macro avg over language pairs). Produces `fig_term_expansion_across_model`. |
| [`by_language_pair/`](by_language_pair/README.md) | Broken out by language pair, one figure per model. Produces `fig_term_expansion_across_languages_{gpt,qwen_3b,qwen_7b}`. |
| [`dictionary/`](dictionary/README.md) | Owns the `build_term_dictionary.py`/`apply_dictionary_to_dev_v1.py` scripts that (re)generate the externally-sourced dictionary term-list variant's data. No experiment-local results — its comparisons run through `shared/scripts/compare_by_model_and_language.py` like the other variants. |

This is a **thin wrapper**: there is no experiment-local data here, only `shared/`, `by_model/`, `by_language_pair/`, and `dictionary/` below. It is a specific comparison drawn from the shared, evolving `shared/data/` → `shared/results/` pipeline documented in the [root README](../../README.md#data), not a separate model run like [`lora_finetuning`](../lora_finetuning/README.md).

The externally-sourced dictionary term-list variant (`shared/results/dev_v1/dictionary/`) has its data-prep scripts in [`dictionary/`](dictionary/README.md); its comparisons run through the shared `compare_by_model_and_language.py` alongside the other variants rather than a standalone comparison script.

For the `dev_v1`-vs-`dev_v2` dataset comparability check (a different axis — not about term-list strategy), see the top-level [`dataset_comparison/`](../dataset_comparison/README.md).

## Layout

| Path | Contents |
|------|----------|
| `shared/scripts/compare_by_model_and_language.py` (`--mode {all,proper_term}`) | Comparison script genuinely shared by **both** axes — each single run writes one output into `by_model/report/` and one into `by_language_pair/report/`. There's no clean way to split it into "the by_model version" and "the by_language_pair version", since both axes need the same underlying comparisons. |
| [`by_model/`](by_model/README.md) | By-model figure, scripts, and report tables. |
| [`by_language_pair/`](by_language_pair/README.md) | By-language-pair figure, scripts, and report tables. |

## Inputs

Both axes read `metrics_summary.json` under `shared/results/dev_v1/` for all 3 models (GPT-4o-mini, Qwen 3B, Qwen 7B) across:

| Variant | Source data | Results path |
| ------- | ------------ | ------------ |
| `original` | `shared/data/dev_v1/dev_v1_original/` | `shared/results/dev_v1/original/few_shot/` |
| `expand` | `shared/data/dev_v1/dev_v1_expand/` | `shared/results/dev_v1/expand/` |
| `cleaned` | `shared/data/dev_v1/dev_v1_cleaned/` | `shared/results/dev_v1/cleaned/` |
| external dictionary | `shared/data/dev_v1/dev_v1_dictionary/` | `shared/results/dev_v1/dictionary/` |

Plus the `no_term` and `random_term` baseline modes — only from `shared/results/dev_v1/original/few_shot/`. These two modes don't depend on the term-list variant, so `expand`/`cleaned`/`dictionary` and `zero_shot` each have only a `proper_term` entry per language in `metrics_summary.json`. `few_shot` is the only `dev_v1/original` variant with all 3 modes, which is why both axes source `original` from there (not `zero_shot`).

`shared/scripts/compare_by_model_and_language.py` handles these `proper_term`-only variants correctly: those sheets simply show `proper_term` rows only, with the mode-label merge computed from the rows actually present rather than assuming a fixed count per mode.
