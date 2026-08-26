# Dictionary term-list variant

Builds an external term dictionary from `dev_v2` and applies it to `dev_v1` as a fourth term-list variant (alongside `original`/`expand`/`cleaned`).

This is a **thin wrapper**: no experiment-local results, only `scripts/` (and `data/` once regenerated). It owns the two scripts that build this variant; the comparisons that use its output run through the parent [`term_expansion/`](../README.md) experiment's shared `compare_by_model_and_language.py --mode proper_term` rather than a standalone script local to this folder.

`data/dev_v2_dictionary/` (the term dictionary `build_term_dictionary.py` builds from dev_v2, then `apply_dictionary_to_dev_v1.py` applies to dev_v1) lives here rather than in the shared `shared/data/` tree since this experiment is its only consumer. It doesn't exist on disk by default — `shared/data/dev_v1/dev_v1_dictionary/` (its already-applied output) does, so the report tables that use it work today without regenerating anything. Regenerating `data/dev_v2_dictionary/` is only needed if you want to rebuild `shared/data/dev_v1/dev_v1_dictionary/` from scratch (e.g. if it were lost) or extend the dictionary to a new language pair.

## Term count

`shared/data/dev_v1/dev_v1_dictionary/` totals **3,732** `proper_terms` entries across all three language pairs (ende 1,351 + enes 1,295 + enru 1,086). This is **additive**, not the dictionary's own size: `apply_dictionary_to_dev_v1.py` merges dictionary matches on top of each sentence's existing `proper_terms` (never replacing them), so 3,732 = the 1,590-term `dev_v1_original` baseline + 2,142 net new matches contributed by the dev_v2-built dictionary. The standalone dictionary itself (`data/dev_v2_dictionary/`) is typically larger still, since not every dictionary entry finds an unambiguous match in a `dev_v1` sentence — but it doesn't exist in this checkout by default (see above), so its own size isn't tracked here.

Recount from the applied output with:

```python
import json, glob
total = sum(
    len(json.loads(line).get("proper_terms", {}))
    for f in glob.glob("shared/data/dev_v1/dev_v1_dictionary/*.jsonl")
    for line in open(f, encoding="utf-8") if line.strip()
)
```

## Reproduce

```
python experiments/term_expansion/dictionary/scripts/build_term_dictionary.py --all
python experiments/term_expansion/dictionary/scripts/apply_dictionary_to_dev_v1.py --all
```

`build_term_dictionary.py` needs `OPENROUTER_API_KEY` in `.env` for its LLM-assisted extraction step (pass `--skip-llm` to seed from human `proper_terms` only, no API calls). Both scripts refuse to overwrite existing output unless passed `--force`.

## Scripts

| File | Role |
|------|------|
| `scripts/dictionary_utils.py` | Shared path constants, I/O, and term-matching/lemmatization helpers used by both scripts below; reuses `shared/lib/data_preparation/openrouter_annotation_common.py` for OpenRouter chat, dotenv loading, and substring matching. |
| `scripts/build_term_dictionary.py` | Builds `data/dev_v2_dictionary/<pair>_term_dictionary.jsonl` from `shared/data/dev_v2/`: seeds from human `proper_terms` (lines 1-1000), then LLM-expands/extracts terms via OpenRouter. |
| `scripts/apply_dictionary_to_dev_v1.py` | Applies the built dictionary to `shared/data/dev_v1/dev_v1_original/`, writing `shared/data/dev_v1/dev_v1_dictionary/<pair>_dev_v1_dictionary.jsonl` — enriches `proper_terms` via reference-based disambiguation, skipping ambiguous matches rather than guessing. |

For the dev_v1-vs-dev_v2 comparability check (a different axis — test set vs. training set, not term-list variant), see [`dataset_comparison/`](../../dataset_comparison/README.md).
