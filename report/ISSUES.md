# Known gaps and open questions

Things about the current repo state worth knowing before writing the paper. Companion to [`README.md`](README.md).

## Dictionary term-list variant has no regeneration path

`experiments/term_expansion/dictionary/` (its README plus `scripts/build_term_dictionary.py` and `scripts/apply_dictionary_to_dev_v1.py`) was deleted when the dictionary variant's comparisons were folded into `experiments/term_expansion/shared/scripts/compare_by_model_and_language.py --mode proper_term`. The scripts are gone, but their *output* is still live and actively used:

- `shared/data/dev_v1/dev_v1_dictionary/` — dev_v1 enriched from the external dictionary.
- `shared/results/dev_v1/dictionary/{gpt,qwen_3b,qwen_7b}/` — evaluation results for that variant, read by `compare_by_model_and_language.py --mode proper_term` and the `by_model`/`by_language_pair` figures/report tables.

There's currently no way to regenerate this data from scratch (e.g. if it were lost) or extend the dictionary to a new language pair — the build (`build_term_dictionary.py`, provenance-aware term dictionary from dev_v2 via OpenRouter) and apply (`apply_dictionary_to_dev_v1.py`) steps no longer exist anywhere in the repo. Either restore a minimal version of those two scripts, or explicitly document `dev_v1_dictionary/`/`shared/results/dev_v1/dictionary/` as frozen/unreproducible data in the paper.

