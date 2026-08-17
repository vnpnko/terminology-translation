# Terminology Translation

Experiments on terminology-constrained machine translation for WMT-style sentence-level data. Code and data accompanying the [`poster/terminology_translation.pdf`](poster/terminology_translation.pdf) poster.

Licensed under [MIT](LICENSE).

Each JSONL record has an English source sentence, a target-language reference, domain terminology, and random control terminology:

```json
{
  "en": "...",
  "de": "...",
  "proper_terms": [{ "term": "translation" }],
  "random_terms": [{ "word": "translation" }]
}
```

## Reproducing the poster figures

Each figure in [`poster/terminology_translation.pdf`](poster/terminology_translation.pdf) is built by a `figure_expN.py` script living in the owning experiment's `scripts/` folder, reading from the shared `results/`/`report/` tree (experiments 01–03) or from a specific experiment folder (05). All four are driven by one shared orchestrator, [`src/analysis/generate_result_figures.py`](src/analysis/generate_result_figures.py):

| Figure | Experiment | Command | Reads from |
| ------ | ---------- | ------- | ---------- |
| [`fig_exp1_term_expansion`](poster/figures/fig_exp1_term_expansion.pdf) | [`01_term_expansion_by_model`](experiments/01_term_expansion_by_model/README.md) | `python src/analysis/generate_result_figures.py --only exp1` | `results/dev_v1/{original,expand,cleaned}/` |
| [`fig_exp23_expansion_strategies`](poster/figures/fig_exp23_expansion_strategies.pdf) | [`02_term_expansion_by_language_pair`](experiments/02_term_expansion_by_language_pair/README.md) | `python src/analysis/generate_result_figures.py --only exp23` | `results/dev_v1/{original,expand,cleaned}/` |
| [`fig_exp4_dev_v1_vs_dev_v2_training`](poster/figures/fig_exp4_dev_v1_vs_dev_v2_training.pdf) | [`03_dataset_comparison`](experiments/03_dataset_comparison/README.md) | `python src/analysis/generate_result_figures.py --only exp4` | `results/dev_v1/original/`, `results/dev_v2/` |
| [`fig_exp5_lora_finetuning`](poster/figures/fig_exp5_lora_finetuning.pdf) | [`05_lora_finetuning`](experiments/05_lora_finetuning/README.md) | `python src/analysis/generate_result_figures.py --only exp5` | `results/dev_v1/original/no-few-shots/`, `experiments/05_lora_finetuning/results/` |

Generate all four at once with `python src/analysis/generate_result_figures.py` (writes to `poster/figures/`).

## Experiments

`experiments/` holds `baseline` plus four numbered, ordered experiments. `01`–`03` are analyses over the shared `data/` → `results/` → `report/` pipeline documented below — each owns a `scripts/` folder with its table/figure-generating code, but no experiment-local data (see their READMEs for exact reproduce commands); `baseline` and `05` are self-contained model-run experiments with their own `data/`/`results/`.

| Experiment | Contents |
| ---------- | -------- |
| [`01_term_expansion_by_model/`](experiments/01_term_expansion_by_model/README.md) | Proper-term expansion: original vs. GPT-expanded vs. domain-filtered, by model. |
| [`02_term_expansion_by_language_pair/`](experiments/02_term_expansion_by_language_pair/README.md) | Same term-expansion strategy comparison, broken out by language pair. |
| [`03_dataset_comparison/`](experiments/03_dataset_comparison/README.md) | `dev_v1` (test) vs. `dev_v2` (training) set comparison, GPT baseline. |
| [`baseline/`](experiments/baseline/README.md) | GPT-4o-mini and Qwen base-model translation notebooks on `dev_v1`. |
| [`05_lora_finetuning/`](experiments/05_lora_finetuning/README.md) | LoRA fine-tuning of Qwen2.5 (3B/7B) vs. GPT-4o-mini and Qwen base, with an Excel export pipeline. |

## Scripts

Genuinely shared code lives in `src/`. Single-experiment scripts live under the owning `experiments/NN_xxx/scripts/` (mirroring `05_lora_finetuning/scripts/`), and import shared `src/` code as `from src.analysis... import ...` / `from src.data_preparation... import ...`.

### Data preparation (shared, `src/data_preparation/`)

| Script                                                     | Purpose                                                                      |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `data_preparation/convert_term_postedits_to_jsonl.py`      | Converts raw SAP `term_postedits` flat files into JSONL mt-task format.      |
| `data_preparation/fill_missing_translations_openrouter.py` | Fills missing target sentences and term translations via OpenRouter.         |
| `data_preparation/annotate_proper_terms_openrouter.py`     | Adds domain `proper_terms` (1–2 IT terms per sentence) via OpenRouter.       |
| `data_preparation/annotate_random_terms_openrouter.py`     | Adds control `random_terms` (non-domain word pairs) after `proper_terms`.    |
| `data_preparation/clean_poor_proper_terms.py`              | Removes weak or generic entries from `proper_terms`.                         |
| `data_preparation/expand_terms.py`                         | Appends additional term pairs to `proper_terms`.                             |
| `data_preparation/collect_term_pairs_from_jsonl.py`        | Aggregates `proper_terms` from v1/v2 dev files into unique term-pair JSONL.  |
| `data_preparation/strip_target_translations.py`            | Clears target sentences and term values while keeping English and term keys. |
| `data_preparation/term_utils.py`                           | Shared path constants and helpers for the dictionary-building scripts below. |

Two dictionary-building scripts are experiment-specific and live in [`experiments/03_dataset_comparison/scripts/`](experiments/03_dataset_comparison/README.md):

| Script | Purpose |
| ------ | ------- |
| `build_term_dictionary.py` | Builds provenance-aware term dictionary from dev_v2 (GPT via OpenRouter). |
| `apply_dictionary_to_dev_v1.py` | Applies the dictionary to dev_v1 into `data/dev_v1/dev_v1_dictionary/` (optional, new files). |

### Analysis (shared, `src/analysis/`)

All analysis scripts require `pandas` and `openpyxl`.

| Script | Purpose |
| ------ | ------- |
| `analysis/metrics_loader.py`, `analysis/figure_common.py`, `analysis/plot_style.py` | Shared metrics-loading and plotting helpers used by every `figure_expN.py`. |
| `analysis/generate_result_figures.py` | Orchestrator: builds all four poster figures by importing each experiment's `figure_expN.py`. |

The table-comparison and figure-generating scripts themselves are experiment-specific; see each experiment's README:

| Script | Location | Purpose |
| ------ | -------- | ------- |
| `compare_models_to_excel.py`, `figure_exp1.py` | [`experiments/01_term_expansion_by_model/scripts/`](experiments/01_term_expansion_by_model/README.md) | Compares GPT, Qwen 3B, and Qwen 7B. Rows grouped by mode. |
| `compare_modes_to_excel.py`, `figure_exp23.py` | [`experiments/02_term_expansion_by_language_pair/scripts/`](experiments/02_term_expansion_by_language_pair/README.md) | Compares `no_term`, `proper_term`, and `random_term`. Rows grouped by model. |
| `compare_datasets_to_excel.py`, `compare_v1_variants_to_excel.py`, `figure_exp4.py` | [`experiments/03_dataset_comparison/scripts/`](experiments/03_dataset_comparison/README.md) | Compares `dev_v1/original` vs `dev_v2`, and `dev_v1/original` vs `dev_v1/dictionary`. |
| `figure_exp5.py` | [`experiments/05_lora_finetuning/scripts/`](experiments/05_lora_finetuning/README.md) | LoRA epoch ablation vs. GPT-4o-mini. |

Example commands:

## Results

All results live in `results/`

Each `gpt/`, `qwen_3b/` and `qwen_7b/` results directory contains a `metrics_summary.json` file.

## Report

Generated comparison Excel files live inside the experiment they compare, under `experiments/<name>/report/` — not in a shared top-level `report/` directory. A table belongs wherever its comparison axis is the experiment's subject: model-vs-model tables belong to the "by model" experiment, mode-vs-mode tables to the "by language pair" experiment, dataset-vs-dataset tables to the dataset-comparison experiment.

| Report dir | Produced by | Naming pattern |
| ---------- | ----------- | --------------- |
| [`experiments/01_term_expansion_by_model/report/`](experiments/01_term_expansion_by_model/README.md#report-tables) | `scripts/compare_models_to_excel.py` | `<dataset>_model_comparison.xlsx` (e.g. `dev_v1_original_model_comparison.xlsx`) |
| [`experiments/02_term_expansion_by_language_pair/report/`](experiments/02_term_expansion_by_language_pair/README.md#report-tables) | `scripts/compare_modes_to_excel.py` | `<dataset>_mode_comparison.xlsx` |
| [`experiments/03_dataset_comparison/report/`](experiments/03_dataset_comparison/README.md#report-tables) | `scripts/compare_datasets_to_excel.py`, `scripts/compare_v1_variants_to_excel.py` | `dev_v1_original_vs_dev_v2_<model>_dataset_comparison.xlsx`, `dev_v1_original_vs_dev_v1_dictionary_gpt_comparison.xlsx` |

See each experiment's README for the exact command to regenerate every table it holds. When a new comparison table doesn't fit any existing experiment, create a new experiment folder for it rather than adding a new top-level report category.

## Data

All data lives in `data/`, grouped by dataset (`dev_v1/`, `dev_v2/`) rather than by processing stage:

| Path | Description |
| ---- | ----------- |
| `dev_v1/dev_v1_original/` | Original course dev set (test set, 500 sentences/language pair). |
| `dev_v1/dev_v1_expand/` | Expanded version of `dev_v1_original/` with additional `proper_terms` (output of `expand_terms.py`). |
| `dev_v1/dev_v1_cleaned/` | Cleaned version of `dev_v1_expand/` with terminology-poor `proper_terms` removed (output of `clean_poor_proper_terms.py`). |
| `dev_v1/dev_v1_dictionary/` | dev_v1 enriched from the term dictionary (output of `apply_dictionary_to_dev_v1.py`, [`experiments/03_dataset_comparison/scripts/`](experiments/03_dataset_comparison/README.md)). |
| `dev_v1/dev_v1_gpt_proposed/` | GPT-proposed term variant. |
| `dev_v2/dev_v2_original/` | Dev set prepared from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus, used as a training-set proxy (see `report/README.md`). |
| `dev_v2/dev_v2_for_training/` | Flat aggregated EN→target term-pair lists from v1/v2 (output of `collect_term_pairs_from_jsonl.py`). |
| `dev_v2/dev_v2_dictionary/` | Term dictionary built from dev_v2 with line IDs, lemmas, and observed inflections (output of `build_term_dictionary.py`, [`experiments/03_dataset_comparison/scripts/`](experiments/03_dataset_comparison/README.md)). |
| `shared_task/` | WMT2025 terminology shared-task materials: task docs (`shared_task_docs/`), track 1 (en→de/es/ru, `shared_task_track1/`), and track 2 (en↔zh, 2015–2024, `shared_task_track2/`). |