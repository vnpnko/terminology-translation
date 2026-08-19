# Term expansion experiments

Four experiments that all compare terminology-constrained translation across different axes of the same `data/` → `results/` → `report/` pipeline: which model does it (`by_model/`), which language pair (`by_language_pair/`), which dataset it's evaluated on (`dataset_comparison/`), and whether the terminology comes from an oracle dictionary or GPT self-proposing it (`gpt_proposed_terms/`). Each sub-experiment is independently reproducible — own `scripts/`, `report/`, and (where applicable) `figures/` — this directory only groups them.

| Experiment | Contents |
| ---------- | -------- |
| [`by_model/`](by_model/README.md) | Proper-term expansion: original vs. GPT-expanded vs. domain-filtered, by model. |
| [`by_language_pair/`](by_language_pair/README.md) | Same term-expansion strategy comparison, broken out by language pair. |
| [`dataset_comparison/`](dataset_comparison/README.md) | `dev_v1` (test) vs. `dev_v2` (training) set comparison, GPT baseline. |
| [`gpt_proposed_terms/`](gpt_proposed_terms/README.md) | GPT-4o-mini `proper_term` (oracle dictionary) vs. `gpt_proposed_term` (GPT self-extracts/proposes terminology), zero-shot on `dev_v1/original`. |
