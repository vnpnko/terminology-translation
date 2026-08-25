# Terminology Translation

Experiments on terminology-constrained machine translation for WMT-style sentence-level data. Code and data supporting an ACL-style paper in progress — see [`report/`](report/README.md) for its outline and current status. An earlier [`poster/terminology_translation.pdf`](poster/terminology_translation.pdf) covers a subset of the same results.

Licensed under [MIT](LICENSE).

Each JSONL record has an English source sentence, a target-language reference, domain terminology, and random control terminology:

```json
{
  "en": "...",
  "de": "...",
  "proper_terms": { "term": "translation" },
  "random_terms": { "word": "translation" }
}
```

## Reproducing the figures

Each figure is built by a `figure_*.py` script living in the owning experiment's `scripts/` folder, reading from the shared `shared/results/`/`report/` tree (the `term_expansion/` experiments) or from a specific experiment folder (`lora_finetuning/`). All are driven by one shared orchestrator, [`shared/lib/analysis/generate_result_figures.py`](shared/lib/analysis/generate_result_figures.py), which writes each figure into its owning experiment's `figures/` directory — that's the figure's canonical home. `poster/figures/` and `report/figures/` are **not** generation targets: copy a figure's home file over by hand whenever it's added or updated. `poster/figures/` is a curated flat collection of whichever figures actually appear in [`poster/terminology_translation.pdf`](poster/terminology_translation.pdf). `report/figures/` instead mirrors the full `experiments/` nesting (one directory per experiment/sub-experiment, dropping the trailing `figures/` — e.g. `experiments/lora_finetuning/epoch_ablation/figures/` → `report/figures/lora_finetuning/epoch_ablation/`) and holds every experiment's figures, both `.pdf` and `.png`, since the paper draws on the full result set rather than a hand-picked subset.

| Figure | Home | Experiment | Command | Reads from |
| ------ | ---- | ---------- | ------- | ---------- |
| [`fig_term_expansion_across_models`](experiments/term_expansion/by_model/figures/fig_term_expansion_across_models.pdf) | `experiments/term_expansion/by_model/figures/` | [`term_expansion/by_model`](experiments/term_expansion/by_model/README.md) | `python shared/lib/analysis/generate_result_figures.py --only by_model` | `shared/results/dev_v1/{original,expand,cleaned}/` |
| [`fig_term_expansion_across_languages_{gpt,qwen_3b,qwen_7b}`](experiments/term_expansion/by_language_pair/figures/fig_term_expansion_across_languages_gpt.pdf) | `experiments/term_expansion/by_language_pair/figures/` | [`term_expansion/by_language_pair`](experiments/term_expansion/by_language_pair/README.md) | `python shared/lib/analysis/generate_result_figures.py --only by_language_pair` | `shared/results/dev_v1/{original,expand,cleaned}/` |
| [`fig_dev_v1_vs_dev_v2_{no_term,proper_term,random_term}`](experiments/dataset_comparison/figures/fig_dev_v1_vs_dev_v2_proper_term.pdf) | `experiments/dataset_comparison/figures/` | [`dataset_comparison`](experiments/dataset_comparison/README.md) | `python shared/lib/analysis/generate_result_figures.py --only dataset_comparison` | `shared/results/dev_v1/original/few_shot/`, `shared/results/dev_v2/` |
| [`fig_lora_epoch_ablation`](experiments/lora_finetuning/epoch_ablation/figures/fig_lora_epoch_ablation.pdf) | `experiments/lora_finetuning/epoch_ablation/figures/` | [`lora_finetuning/epoch_ablation`](experiments/lora_finetuning/epoch_ablation/README.md) | `python shared/lib/analysis/generate_result_figures.py --only epoch_ablation` | `shared/results/dev_v1/original/zero_shot/`, `experiments/lora_finetuning/shared/results/` |
| [`fig_lora_best_models`](experiments/lora_finetuning/best_models/figures/fig_lora_best_models.pdf) | `experiments/lora_finetuning/best_models/figures/` | [`lora_finetuning/best_models`](experiments/lora_finetuning/best_models/README.md) | `python shared/lib/analysis/generate_result_figures.py --only best_models` | `experiments/lora_finetuning/shared/results/`, `experiments/lora_finetuning/shared/run_registry.json` |

Generate all at once with `python shared/lib/analysis/generate_result_figures.py` (writes each into its home dir above; `dataset_comparison` writes 3 files, one per term mode, and `by_language_pair` writes 3 files, one per model). Of these, `fig_term_expansion_across_models`, the 3 `fig_term_expansion_across_languages_{model}` figures, `fig_lora_epoch_ablation`, and `fig_lora_best_models` are copied to `poster/figures/` for the poster. Every experiment's figures are copied to `report/figures/` for the paper, under a matching `report/figures/<experiment>/<sub-experiment>/` directory.

## Notebooks

[`shared/notebooks/`](shared/notebooks/) holds the three manually-run model notebooks (`gpt.ipynb`, `qwen_base.ipynb`, `qwen_finetuned.ipynb`) used to produce the `lora_finetuning` results — run individually (e.g. on LRZ AI Systems), not driven by any script.

## Experiments

`experiments/` groups experiments into three top-level directories: `dataset_comparison/`, `term_expansion/`, and `lora_finetuning/` (5 nested sub-experiments). Each experiment owns a `scripts/` folder with its table/figure-generating code (see their READMEs for exact reproduce commands); `lora_finetuning/` is a self-contained model-run experiment with its own `data/`/`results/`.

| Experiment | Contents |
| ---------- | -------- |
| [`dataset_comparison/`](experiments/dataset_comparison/README.md) | `dev_v1` vs. `dev_v2` terminology dataset comparison, per model. |
| [`term_expansion/`](experiments/term_expansion/README.md) | Proper-term expansion: original vs. GPT-expanded vs. GPT-cleaned (domain-filtered) vs. externally-sourced dictionary, both aggregated by model and broken out by language pair. |
| [`lora_finetuning/`](experiments/lora_finetuning/README.md) | LoRA fine-tuning of Qwen2.5 (3B/7B) vs. GPT and Qwen base: 5 sub-experiments (`epoch_ablation/`, `best_models/`, `base_vs_lora/`, `few_shot_ablation/`, `leakage_check/`) sharing `shared/results/`, `shared/data/`, and `shared/run_registry.json`. |

## Scripts

Genuinely shared code lives in `shared/lib/`. Single-experiment scripts live under the owning experiment's `scripts/` folder (e.g. `experiments/lora_finetuning/shared/scripts/`, `experiments/term_expansion/by_model/scripts/`), and import shared `shared/lib/` code as `from shared.lib.analysis... import ...` / `from shared.lib.data_preparation... import ...` — this holds regardless of how deeply an experiment is nested under `experiments/`.

### Data preparation (shared, `shared/lib/data_preparation/`)

| Script                                                     | Purpose                                                                      |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `data_preparation/convert_term_postedits_to_jsonl.py`      | Converts raw SAP `term_postedits` flat files into JSONL mt-task format.      |
| `data_preparation/fill_missing_translations_openrouter.py` | Fills missing target sentences and term translations via OpenRouter.         |
| `data_preparation/annotate_proper_terms_openrouter.py`     | Adds domain `proper_terms` (1–2 IT terms per sentence) via OpenRouter.       |
| `data_preparation/annotate_random_terms_openrouter.py`     | Adds control `random_terms` (non-domain word pairs) after `proper_terms`.    |
| `data_preparation/clean_poor_proper_terms.py`              | Removes weak or generic entries from `proper_terms`.                         |
| `data_preparation/expand_terms.py`                         | Appends additional term pairs to `proper_terms`.                             |
| `data_preparation/strip_target_translations.py`            | Clears target sentences and term values while keeping English and term keys. |
| `data_preparation/check_duplicate_sources.py`               | Report-only: checks a JSONL file for duplicate source (`en`) sentences.      |

### Analysis (shared, `shared/lib/analysis/`)

All analysis scripts require `pandas` and `openpyxl`.

| Script | Purpose |
| ------ | ------- |
| `analysis/metrics_loader.py`, `analysis/grouped_bar_figure_common.py`, `analysis/plot_style.py` | Shared metrics-loading and plotting helpers used by every `figure_expN.py`. |
| `analysis/generate_result_figures.py` | Orchestrator: builds all result figures (poster and report) by importing each experiment's `figure_expN.py`. |

The table-comparison and figure-generating scripts themselves are experiment-specific; see each experiment's README:

| Script | Location | Purpose |
| ------ | -------- | ------- |
| `compare_by_model_and_language.py` (`--mode {all,proper_term}`) | [`experiments/term_expansion/shared/scripts/`](experiments/term_expansion/README.md) | Compares GPT, Qwen 3B, and Qwen 7B, and `ende`/`enru`/`enes` — writes both axes' report tables in one run. |
| `figure_by_model.py` | [`experiments/term_expansion/by_model/scripts/`](experiments/term_expansion/by_model/README.md) | Aggregated across language pairs, grouped by model. |
| `figure_by_language_pair.py` | [`experiments/term_expansion/by_language_pair/scripts/`](experiments/term_expansion/by_language_pair/README.md) | One figure per model, grouped by language pair. |
| `compare_datasets_to_excel.py`, `figure_dataset_comparison.py` | [`experiments/dataset_comparison/scripts/`](experiments/dataset_comparison/README.md) | Compares `dev_v1/original` vs `dev_v2`. |
| `figure_epoch_ablation.py` | [`experiments/lora_finetuning/epoch_ablation/scripts/`](experiments/lora_finetuning/epoch_ablation/README.md) | LoRA epoch ablation, both model sizes. |
| `figure_best_models.py` | [`experiments/lora_finetuning/best_models/scripts/`](experiments/lora_finetuning/best_models/README.md) | Best LoRA config vs. GPT. |

## Results

All results live in `shared/results/`, grouped by dataset/variant then by model (`gpt/`, `qwen_3b/`, `qwen_7b/`). Each model directory holds one `metrics_summary.json` plus one flat prediction file per language and mode (`{lang}_..._{mode}_predictions.jsonl` — no per-language subdirectory). `shared/results/dev_v1/original/` additionally splits into `zero_shot/` and `few_shot/` subdirectories, since that's the one dataset evaluated both ways; `dev_v1/{expand,cleaned,dictionary}/` each have a single, unqualified model directory. `shared/results/dev_v2/` is flat like the other unqualified variants (`gpt/`, `qwen_3b/`, `qwen_7b/`); it's used both as an eval target and, via `dataset_comparison`, compared directly against `dev_v1/original/`.

## Report

Generated comparison Excel files live inside the experiment they compare, under `experiments/<name>/report/` — not in a shared top-level `report/` directory. A table belongs wherever its comparison axis is the experiment's subject: model-vs-model and language-vs-language tables belong to the `term_expansion` experiment, dataset-vs-dataset tables to the dataset-comparison experiment.

| Report dir | Produced by | Naming pattern |
| ---------- | ----------- | --------------- |
| [`experiments/term_expansion/by_model/report/`](experiments/term_expansion/by_model/README.md#report-tables), [`by_language_pair/report/`](experiments/term_expansion/by_language_pair/README.md#report-tables) | `shared/scripts/compare_by_model_and_language.py` (default `--mode all`) | `model_comparison.xlsx` (by_model) + `language_comparison.xlsx` (by_language_pair), one sheet per dataset variant: `dev_v1`, `dev_v2` |
| [`experiments/term_expansion/by_model/report/`](experiments/term_expansion/by_model/README.md#report-tables), [`by_language_pair/report/`](experiments/term_expansion/by_language_pair/README.md#report-tables) | `shared/scripts/compare_by_model_and_language.py --mode proper_term` | `proper_term_across_models.xlsx` (by_model) + `proper_term_across_languages.xlsx` (by_language_pair), rows: 4 term-list variants × language/model |
| [`experiments/dataset_comparison/report/`](experiments/dataset_comparison/README.md#report-table) | `scripts/compare_datasets_to_excel.py` | `dataset_comparison.xlsx` (one sheet per model) |

See each experiment's README for the exact command to regenerate every table it holds. When a new comparison table doesn't fit any existing experiment, create a new experiment folder for it rather than adding a new top-level report category.

## Data

All data lives in `shared/data/`, grouped by dataset (`dev_v1/`, `dev_v2/`) rather than by processing stage.

| Path | Description |
| ---- | ----------- |
| `dev_v1/dev_v1_original/` | Original course dev set (test set, 500 sentences/language pair). |
| `dev_v1/dev_v1_expand/` | Expanded version of `dev_v1_original/` with additional `proper_terms` (output of `expand_terms.py`). |
| `dev_v1/dev_v1_cleaned/` | Cleaned version of `dev_v1_expand/` with terminology-poor `proper_terms` removed (output of `clean_poor_proper_terms.py`). |
| `dev_v1/dev_v1_dictionary/` | dev_v1 enriched from an external term dictionary built from dev_v2. Regenerate via [`experiments/term_expansion/dictionary/`](experiments/term_expansion/dictionary/README.md)'s `build_term_dictionary.py`/`apply_dictionary_to_dev_v1.py`. |
| `dev_v2/` | Dev set prepared from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus, used as a training-set proxy (see `report/README.md`). The post-removal, few-shot-trimmed training set derived from this lives at [`experiments/lora_finetuning/shared/data/training/`](experiments/lora_finetuning/README.md). |
| `shared_task/` | WMT2025 terminology shared-task materials: task docs (`shared_task_docs/`), track 1 (en→de/es/ru, `shared_task_track1/`), and track 2 (en↔zh, 2015–2024, `shared_task_track2/`). |