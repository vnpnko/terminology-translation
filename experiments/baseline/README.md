# Baseline

Zero-shot baselines on `dev_v1`: GPT-4o-mini (via OpenRouter) and a Qwen2.5 base model, each translated under all three terminology modes (`no_term`, `proper_term`, `random_term`).

## Layout

| Path | Contents |
|------|----------|
| `notebooks/openai_baseline.ipynb` | GPT-4o-mini baseline via OpenRouter |
| `notebooks/qwen_baseline.ipynb` | Qwen2.5 base model baseline |
| `results/openai_translation/` | GPT predictions (`{lang}_..._{mode}_predictions.jsonl`) + `metrics_summary.json` |
| `results/qwen_translation/` | Qwen predictions (`{lang}_..._{mode}_predictions.jsonl`) + `metrics_summary.json` |

## Running

Both notebooks auto-detect input data from `data/dev_v1/dev_v1_original/`, `data/dev_v1/dev_v1_expand/`, `data/dev_v1/dev_v1_cleaned/` (repo root), `experiments/baseline/`, or the notebook's working directory — set `DATA_DIR` explicitly if detection fails.

- `notebooks/openai_baseline.ipynb` requires `OPENROUTER_API_KEY` (see root [`.env.example`](../../.env.example)). No standalone CLI equivalent exists in this repo — run the notebook directly.
- `notebooks/qwen_baseline.ipynb` requires a GPU (LRZ/Colab) and an optional `HF_TOKEN` env var for gated models.

Each notebook writes predictions per language/mode plus a `metrics_summary.json` (BLEU, chrF, terminology accuracy/consistency) under `results/`.
