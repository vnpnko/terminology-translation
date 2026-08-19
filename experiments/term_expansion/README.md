# Term expansion experiments

Three experiments that all compare terminology-constrained translation across different axes of the same `data/` → `results/` → `report/` pipeline: which model does it (`by_model/`), which language pair (`by_language_pair/`), and whether the term list comes from an externally built dictionary (`dictionary/`). Each sub-experiment is independently reproducible — own `scripts/` and (where applicable) `report/`/`figures/`/`data/` — this directory only groups them. `dictionary/` is data-prep-only (no `report/` of its own — its comparisons live in `by_model/` and `by_language_pair/`, see below).

| Experiment | Contents |
| ---------- | -------- |
| [`by_model/`](by_model/README.md) | Proper-term expansion: original vs. GPT-expanded vs. domain-filtered, by model. |
| [`by_language_pair/`](by_language_pair/README.md) | Same term-expansion strategy comparison, broken out by language pair. |
| [`dictionary/`](dictionary/README.md) | Builds the externally-sourced dictionary term-list variant (data prep only — comparisons against it live in `by_model/` and `by_language_pair/`). |

For the `dev_v1`-vs-`dev_v2` dataset comparability check (a different axis — not about term-list strategy), see the top-level [`dataset_comparison/`](../dataset_comparison/README.md).
