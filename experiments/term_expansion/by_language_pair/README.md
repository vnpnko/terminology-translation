# Term expansion by language pair

Broken out by language pair (`ende` / `enru` / `enes`), from two angles: GPT-4o-mini alone, and a direct Qwen2.5-3B-vs-7B comparison. Reuses the parent [`term_expansion/`](../README.md)'s shared `shared/data/` → `shared/results/` pipeline — see that README for the Inputs table.

| Sub-experiment | Contents |
| -------------- | -------- |
| [`gpt/`](gpt/README.md) | GPT-4o-mini only. Produces `fig_term_expansion_across_languages_gpt`, used in the report's main body (Figure 1). |
| [`qwen/`](qwen/README.md) | Qwen2.5-3B vs. 7B, compared directly via stacked bars. Produces `fig_term_expansion_qwen_size_stacked`, used in the report's Appendix. |

`shared/scripts/by_language_pair_common.py` holds the data-loading and constants both figure scripts reuse (they compare the same term-list variants by language pair, differing only in which baseline model(s) they plot). `shared/scripts/compare_by_model_and_language.py` is the comparison script also shared with the parent [`by_model/`](../by_model/README.md) sub-experiment — each run writes one output here and one into `by_model/report/`.

## Report tables

| File | Regenerate |
| ---- | ---------- |
| `report/language_comparison.xlsx` (sheets: `dev_v1`, `dev_v2`; rows grouped by mode then model, columns per language) | `python experiments/term_expansion/by_language_pair/shared/scripts/compare_by_model_and_language.py` (also writes the sibling `../by_model/report/model_comparison.xlsx` in the same run) |
| `report/proper_term_across_languages.xlsx` (rows: 4 term-list variants x model, columns per language) | `python experiments/term_expansion/by_language_pair/shared/scripts/compare_by_model_and_language.py --mode proper_term` (also writes the sibling `../by_model/report/proper_term_across_models.xlsx` in the same run) |

`language_comparison.xlsx` does not include `dev_v1_expand`/`dev_v1_cleaned`/`dev_v1_dictionary` sheets, since those variants only have `proper_term` data, which already appears in `proper_term_across_languages.xlsx`.
