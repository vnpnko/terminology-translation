# Term expansion experiments

Three experiments that all compare terminology-constrained translation across different axes of the same `data/` → `results/` → `report/` pipeline: which model does it (`by_model/`), which language pair (`by_language_pair/`), and whether the term list comes from an externally built dictionary (`dictionary/`). Each sub-experiment is independently reproducible — own `scripts/`, `report/`, and (where applicable) `figures/`/`data/` — this directory only groups them.

`gpt_proposed_terms/` is grouped here too: a related but distinct question (can GPT self-propose terminology as well as an oracle dictionary?) evaluated on the same `dev_v1/original` zero-shot GPT baseline.

| Experiment | Contents |
| ---------- | -------- |
| [`by_model/`](by_model/README.md) | Proper-term expansion: original vs. GPT-expanded vs. domain-filtered, by model. |
| [`by_language_pair/`](by_language_pair/README.md) | Same term-expansion strategy comparison, broken out by language pair. |
| [`dictionary/`](dictionary/README.md) | `dev_v1/original` vs. `dev_v1/dictionary` (externally built term-list variant), GPT baseline. |
| [`gpt_proposed_terms/`](gpt_proposed_terms/README.md) | GPT-4o-mini `proper_term` (oracle dictionary) vs. `gpt_proposed_term` (GPT self-extracts/proposes terminology), zero-shot on `dev_v1/original`. |

For the `dev_v1`-vs-`dev_v2` dataset comparability check (a different axis — not about term-list strategy), see the top-level [`dataset_comparison/`](../dataset_comparison/README.md).
