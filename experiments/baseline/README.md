# Baseline

Zero-shot GPT-4o-mini baseline on `dev_v1` (via OpenRouter), translated under all three terminology modes (`no_term`, `proper_term`, `random_term`).

## Layout

| Path | Contents |
|------|----------|
| `openai_baseline.ipynb` | GPT-4o-mini baseline via OpenRouter |
| `results/` | GPT predictions (`{lang}_..._{mode}_predictions.jsonl`) + `metrics_summary.json` |

## Running

The notebook auto-detects input data from `data/dev_v1/dev_v1_original/`, `data/dev_v1/dev_v1_expand/`, `data/dev_v1/dev_v1_cleaned/` (repo root), `experiments/baseline/`, or the notebook's working directory — set `DATA_DIR` explicitly if detection fails.

`openai_baseline.ipynb` requires `OPENROUTER_API_KEY` (see root [`.env.example`](../../.env.example)). No standalone CLI equivalent exists in this repo — run the notebook directly.

The notebook writes predictions per language/mode plus a `metrics_summary.json` (BLEU, chrF, terminology accuracy/consistency) under `results/`.
