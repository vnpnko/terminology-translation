# Terminology Translation

Experiments on terminology-constrained machine translation for WMT-style sentence-level data. Each example contains an English source sentence, a target-language reference, domain terminology, and random control terminology.

## Data Format

One JSON object per line:

```json
{
  "en": "In such cases you may use the Move Items or Merge feature.\n",
  "de": "In solchen Fällen können Sie die Funktion Elemente verschieben oder Zusammenführen verwenden.\n",
  "proper_terms": {
    "item": "Element"
  },
  "random_terms": {
    "Move": "Verschieben"
  }
}
```

| Field | Description |
|---|---|
| `en` | English source sentence. |
| `de` / `es` / `ru` | Target-language reference translation. |
| `proper_terms` | `{english_term: target_translation}` for domain/SAP/IT terminology. |
| `random_terms` | `{english_word: target_translation}` for ordinary control vocabulary. |

The JSONL format is unchanged across all evaluators and data-preparation scripts.

## Translation Modes

| Mode | Prompt terminology | Measures |
|---|---|---|
| `no_term` | No terminology block. | Unconstrained baseline MT quality. |
| `proper_term` | Uses `proper_terms`. | Domain terminology-constrained MT. |
| `random_term` | Uses `random_terms` after removing keys that overlap `proper_terms`. | Control condition for non-domain glossary hints. |

## Project Layout

```text
terminology-translation/
├── src/
│   └── mt_eval/
│       ├── __init__.py
│       └── core.py
├── scripts/
│   ├── Ivan/
│   │   ├── run_eval.py
│   │   ├── fill_proper_terms.py
│   │   ├── fill_random_terms.py
│   │   └── fill_translations.py
│   └── Robin/
│       ├── run_eval.py
│       └── build_term_jsonl.py
├── Baseline/
│   ├── ende_dev_v2.jsonl
│   └── openrouter_outputs/
├── data/
└── README.md
```

Shared loading, prompt construction, mode handling, BLEU, chrF, terminology accuracy, and terminology consistency live in `src/mt_eval/core.py`.

## Install

```bash
pip install openai sacrebleu tqdm
```

For Robin's local Qwen evaluator:

```bash
pip install transformers accelerate torch sacrebleu tqdm
```

## Ivan Evaluator: OpenRouter / GPT-4o-mini

Set `OPENROUTER_API_KEY` in `scripts/.env` or pass `--api-key`.

```bash
python scripts/Ivan/run_eval.py --data-file Baseline/ende_dev_v2.jsonl
```

Optional mode selection:

```bash
python scripts/Ivan/run_eval.py \
  --data-file Baseline/ende_dev_v2.jsonl \
  --modes no_term proper_term random_term
```

Outputs keep the existing names and structure:

```text
Baseline/openrouter_outputs/<data_stem>_<mode>_predictions.jsonl
Baseline/openrouter_outputs/<data_stem>_metrics_summary.json
```

## Robin Evaluator: Local Qwen

The runner loads `Qwen/Qwen2.5-Coder-3B-Instruct` with Hugging Face Transformers.

```bash
python scripts/Robin/run_eval.py --data-dir data/
```

Optional mode selection:

```bash
python scripts/Robin/run_eval.py \
  --data-dir data/ \
  --modes no_term proper_term random_term
```

Robin's evaluator scans `data/*.jsonl` and keeps files ending in `_dev.jsonl`. It writes structured prediction and metrics files using the same schema as Ivan's runner.

## Metrics

The shared evaluator preserves the existing metric behavior:

- `compute_bleu_chrf(hyps, refs)` calls `sacrebleu.corpus_bleu(hyps, [refs])` and `sacrebleu.corpus_chrf(hyps, [refs])` with default settings.
- `terminology_accuracy_advanced(samples, predictions, mode)` computes the existing source-count to target-count ratio per term.
- `terminology_consistency_advanced(samples, predictions, mode)` computes the existing pseudo-reference consistency scores.

The metrics JSON keys are unchanged.

## Adding A New Model

Create or copy a runner and replace only its `translate_sample()` function. Keep the runner calling shared helpers from `src/mt_eval/core.py`:

- `build_translation_prompt(source_text, target_lang, terminology)`
- `terminology_for_mode(sample, mode)`
- `strip_output_tags(text)`
- `compute_bleu_chrf(hyps, refs)`
- `terminology_accuracy_advanced(samples, predictions, mode)`
- `terminology_consistency_advanced(samples, predictions, mode)`

## Data Preparation

Existing data-preparation scripts remain unchanged:

```bash
python scripts/Ivan/fill_proper_terms.py -t de -i input.jsonl -o output.jsonl
python scripts/Ivan/fill_random_terms.py -t de -i input.jsonl -o output.jsonl
python scripts/Ivan/fill_translations.py ende_dev_1.jsonl
python scripts/Robin/build_term_jsonl.py -t de --data-dir "data/asap data v2/orig data/ende"
```

