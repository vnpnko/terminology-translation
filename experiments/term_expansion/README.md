# Term expansion experiments

Two experiments that compare terminology-constrained translation across different axes of the same `data/` → `results/` → `report/` pipeline: which model does it (`by_model/`) and which language pair (`by_language_pair/`). Each sub-experiment is independently reproducible — own `scripts/`, `report/`, `figures/`, `data/` — this directory only groups them.

| Experiment | Contents |
| ---------- | -------- |
| [`by_model/`](by_model/README.md) | Proper-term expansion: original vs. GPT-expanded vs. domain-filtered vs. externally-sourced dictionary, by model. |
| [`by_language_pair/`](by_language_pair/README.md) | Same term-expansion strategy comparison, broken out by language pair. |

The externally-sourced dictionary term-list variant (`results/dev_v1/dictionary/`) no longer has its own experiment folder — its data-prep scripts and standalone comparisons were merged into `by_model/` and `by_language_pair/`.

For the `dev_v1`-vs-`dev_v2` dataset comparability check (a different axis — not about term-list strategy), see the top-level [`dataset_comparison/`](../dataset_comparison/README.md).
