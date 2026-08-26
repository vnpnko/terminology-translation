# Term expansion

Compares terminology-constrained translation across `no_term`, `random_term`, the three `proper_term` source variants (original / GPT-expanded / GPT-cleaned), and the external term dictionary, from two angles: aggregated by model, and broken out by language pair.

| Sub-experiment | Contents |
| -------------- | -------- |
| [`by_model/`](by_model/README.md) | Aggregated by model (macro avg over language pairs). Produces `fig_term_expansion_across_models`. |
| [`by_language_pair/`](by_language_pair/README.md) | Broken out by language pair — GPT-4o-mini alone (`gpt/`), and a direct Qwen2.5-3B-vs-7B comparison (`qwen/`). |
| [`dictionary/`](dictionary/README.md) | Owns the `build_term_dictionary.py`/`apply_dictionary_to_dev_v1.py` scripts that (re)generate the externally-sourced dictionary term-list variant's data. No experiment-local results — its comparisons run through `by_language_pair/shared/scripts/compare_by_model_and_language.py` like the other variants. |
| [`gpt_proposed_term_pipeline/`](gpt_proposed_term_pipeline/README.md) | GPT-4o-mini's oracle `proper_term` mode vs. `gpt_proposed_term` mode (GPT extracts and proposes its own terminology, no oracle access), zero-shot, `dev_v1/original`. Self-contained results (own `results/`/`report/`, not the shared `dev_v1/` tree) — its output isn't a real term-list variant. Recently restored/ported; see its README for caveats before relying on the numbers. |

This is a **thin wrapper**: there is no experiment-local data here, only `by_model/`, `by_language_pair/`, and `dictionary/` below. It is a specific comparison drawn from the shared, evolving `shared/data/` → `shared/results/` pipeline documented in the [root README](../../README.md#data), not a separate model run like [`lora_finetuning`](../lora_finetuning/README.md).

The externally-sourced dictionary term-list variant (`shared/results/dev_v1/dictionary/`) has its data-prep scripts in [`dictionary/`](dictionary/README.md); its comparisons run through the shared `compare_by_model_and_language.py` alongside the other variants rather than a standalone comparison script.

For the `dev_v1`-vs-`dev_v2` dataset comparability check (a different axis — not about term-list strategy), see the top-level [`dataset_comparison/`](../dataset_comparison/README.md).

## Layout

| Path | Contents |
|------|----------|
| [`by_model/`](by_model/README.md) | By-model figure, scripts, and report tables. |
| [`by_language_pair/`](by_language_pair/README.md) | By-language-pair figures (`gpt/`, `qwen/`), their shared data-loading code (`shared/scripts/by_language_pair_common.py`), the comparison script also shared with `by_model/` (`shared/scripts/compare_by_model_and_language.py`), and report tables. |

## Inputs

Both axes read `metrics_summary.json` under `shared/results/dev_v1/` for all 3 models (GPT, Qwen 3B, Qwen 7B) across:

| Variant | Terms | Source data | Results path |
| ------- | ----: | ------------ | ------------ |
| `original` | 1,590 | `shared/data/dev_v1/dev_v1_original/` | `shared/results/dev_v1/original/few_shot/` |
| `expand` | 2,573 | `shared/data/dev_v1/dev_v1_expand/` | `shared/results/dev_v1/expand/` |
| `cleaned` | 1,223 | `shared/data/dev_v1/dev_v1_cleaned/` | `shared/results/dev_v1/cleaned/` |
| external dictionary | 3,732 | `shared/data/dev_v1/dev_v1_dictionary/` | `shared/results/dev_v1/dictionary/` |

Term counts are `proper_terms` entries summed across all three language pairs (see [`dictionary/README.md`](dictionary/README.md#term-count) for the recount command). `cleaned` filters `expand`'s list for domain-specificity, so it falls below even `original`'s count, not just below `expand`'s. `external dictionary` is additive on top of `original` (1,590 + 2,142 net new matches from a dev_v2-built dictionary), not a from-scratch count.

Plus the `no_term` and `random_term` baseline modes — only from `shared/results/dev_v1/original/few_shot/`. These two modes don't depend on the term-list variant, so `expand`/`cleaned`/`dictionary` and `zero_shot` each have only a `proper_term` entry per language in `metrics_summary.json`. `few_shot` is the only `dev_v1/original` variant with all 3 modes, which is why both axes source `original` from there (not `zero_shot`).

`by_language_pair/shared/scripts/compare_by_model_and_language.py` handles these `proper_term`-only variants correctly: those sheets simply show `proper_term` rows only, with the mode-label merge computed from the rows actually present rather than assuming a fixed count per mode.

Report tables (the two cross-model `.xlsx` comparisons) live at [`by_language_pair/report/`](by_language_pair/report/) — see [`by_language_pair/README.md`](by_language_pair/README.md#report-tables) for regeneration commands.
