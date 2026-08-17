# Report planning

This directory will hold the ACL-style paper for the project (template: [ACL conference LaTeX template](https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj)). Before writing prose, this doc plans the **Experiments section only** — which experiments exist, where their data/results live, and how documented each one currently is. The rest of the ACL skeleton (Related Work, Ethics Statement, Limitations, Appendix, etc.) is deferred to a later pass.

Companion doc: [`ISSUES.md`](ISSUES.md) — every messy/undocumented/broken thing found while building this plan, with suggested fixes or open questions.

## Canonical vocabulary

Every experiment below draws on the same building blocks (source: [`src/analysis/metrics_loader.py`](../src/analysis/metrics_loader.py)). Use these exact names in the paper so tables are consistent:

| Axis | Values |
| ---- | ------ |
| Modes | `no_term`, `proper_term`, `random_term` (control) |
| Models | GPT-4o-mini (`gpt`), Qwen2.5-3B (`qwen_3b`), Qwen2.5-7B (`qwen_7b`) |
| Language pairs | EN→DE (`ende`), EN→RU (`enru`), EN→ES (`enes`) |
| Metrics | BLEU, chrF, Term Accuracy % (`terminology_accuracy.avg_ratio_pct`), Macro Consistency, Weighted Consistency |
| dev_v1 term-list variants | `original` (1–2 gold terms/sentence), `expand` (GPT-suggested contextual expansion), `cleaned` (GPT-expand, domain-filtered), `dictionary` (external dictionary built from dev_v2) |

**Datasets** (poster definitions, see [`poster/terminology_translation.pdf`](../poster/terminology_translation.pdf)):
- **dev_v1** — 500 sentences/language pair, the held-out test set.
- **dev_v2** — sourced from SAP `term_postedits`; **not** the original shared-task dev set, used as a proxy because it's the same documentation domain. Originally 2000 lines/language; see §3.4.2 for how it was cut down for training.

## Experiments section outline

Status legend: 🟢 Documented · 🟡 Partially documented · 🔴 Undocumented · ⚫ Data/results missing.

### 3.1 Terminology modes across language pairs (no-term / proper-term / random-term, EN→DE / EN→RU / EN→ES) — 🟢
- **RQ:** Does injecting explicit terminology constraints improve BLEU/chrF/term accuracy over no constraints, does an unrelated (random-term) constraint list act as a fair control, and does that benefit vary by language pair? How does an external dictionary (built from dev_v2) compare, in this same mode×language breakdown?
- **Repo location:** [`experiments/02_term_expansion_by_language_pair/`](../experiments/02_term_expansion_by_language_pair/README.md) (modes) and [`experiments/03_dataset_comparison/`](../experiments/03_dataset_comparison/README.md) (dictionary variant)
- **Scripts:** [`experiments/02_term_expansion_by_language_pair/scripts/compare_languages_to_excel.py`](../experiments/02_term_expansion_by_language_pair/scripts/compare_languages_to_excel.py), [`experiments/03_dataset_comparison/scripts/compare_v1_variants_to_excel.py`](../experiments/03_dataset_comparison/scripts/compare_v1_variants_to_excel.py)
- **Tables:** `experiments/02_term_expansion_by_language_pair/report/dev_v1_{original,expand,cleaned}_language_comparison.xlsx`, `dev_v2_language_comparison.xlsx`; `experiments/03_dataset_comparison/report/dev_v1_original_vs_dev_v1_dictionary_gpt_comparison.xlsx`
- **Figure:** `poster/figures/fig_expansion_strategies.pdf`
- **Poster finding to reuse:** EN→ES strongest on both BLEU and term accuracy; EN→DE beats EN→RU on BLEU, EN→RU leads on term accuracy.
- **Known gap (see `03_dataset_comparison/README.md`):** `poster/figures/fig_dictionary_vs_original.pdf` exists but no current script regenerates it — treat as unreproducible until fixed, don't cite it as if it were live.

### 3.2 Terminology modes across models (GPT-4o-mini vs Qwen 3B vs Qwen 7B) — 🟢 (see ISSUES §4 for workbook duplication)
- **RQ:** How do a closed frontier model and two open base models compare on terminology-constrained translation, and does that comparison hold across term-list variants (original / GPT-expand / GPT-clean)?
- **Repo location:** [`experiments/01_term_expansion_by_model/`](../experiments/01_term_expansion_by_model/README.md)
- **Script:** [`experiments/01_term_expansion_by_model/scripts/compare_models_to_excel.py`](../experiments/01_term_expansion_by_model/scripts/compare_models_to_excel.py)
- **Tables:** `experiments/01_term_expansion_by_model/report/dev_v1_{original,expand,cleaned}_model_comparison.xlsx`, `dev_v2_model_comparison.xlsx` — **use only the `metrics` sheet** in `dev_v1_original_model_comparison.xlsx` (see ISSUES §4, `Sheet1` and `terms expansion types` are redundant near-duplicates)
- **Figure:** `poster/figures/fig_term_expansion.pdf`

### 3.3 dev_v1 vs dev_v2 dataset comparison — 🟢 (background context, not one of the required axes, but needed to set up §3.4.2)
- **RQ:** How comparable is the dev_v2 proxy training set to the dev_v1 test set on GPT-4o-mini?
- **Repo location:** [`experiments/03_dataset_comparison/`](../experiments/03_dataset_comparison/README.md)
- **Script:** [`experiments/03_dataset_comparison/scripts/compare_datasets_to_excel.py`](../experiments/03_dataset_comparison/scripts/compare_datasets_to_excel.py)

### 3.4 LoRA fine-tuning (Qwen2.5-3B / 7B, 1–3 epochs) — 🟡 README's export section is stale
- **RQ:** Does LoRA fine-tuning on dev_v2 close the gap between open Qwen models and GPT-4o-mini on terminology-constrained translation?
- **Repo location:** [`experiments/04_lora_finetuning/`](../experiments/04_lora_finetuning/README.md), evaluated on `proper_term` mode only.
- **Canonical runs** (from [`scripts/run_registry.json`](../experiments/04_lora_finetuning/scripts/run_registry.json)): `base` (no fine-tuning), `lora_1ep_fs` (1 epoch, few-shot), `lora_2ep_nofs` (2 epochs, no few-shot), `lora_3ep_nofs` (3 epochs, no few-shot), for both 3B and 7B, plus `gpt_base`.
- **Tables:** the README documents `results_Qwen2.5-{3B,7B}.xlsx` and `results_GPT4o-mini.xlsx` as the export output — **none of these three files exist in the repo.** The only workbook actually committed is `comparisons.xlsx` (hand-assembled, differently structured, sheet names not matching the README's documented sheet list). `run_registry.json`'s own `workbook` fields still point at the three missing files — the registry is stale evidence of an abandoned export pipeline. Use `comparisons.xlsx` as the real source of truth for now, but re-run/rebuild the export before citing numbers in the paper (see ISSUES §2).
- **Figure:** `poster/figures/fig_lora_finetuning.pdf` — epoch ablation vs. GPT-4o-mini.
- **Poster findings to reuse:** LoRA gains build through epochs, peak ~2 epochs; 7B term accuracy stable, 3B term accuracy drops without few-shot but recovers after fine-tuning; best 7B LoRA closes much of the BLEU gap to GPT, especially EN→RU, while GPT few-shot stays strongest on term accuracy.

#### 3.4.1 Few-shot ablation — 🟡 two separate, differently-scoped experiments exist under this name
There are **two distinct few-shot comparisons** in the repo; the report needs to pick which one(s) it reports and name them so readers don't conflate them:

1. **Baseline-level few-shot** (GPT + Qwen 3B/7B, no fine-tuning): `results/dev_v1/original/no-few-shots/{gpt,qwen_3b,qwen_7b}/` vs `results/dev_v1/original/with-few-shots/{gpt,qwen_3b,qwen_7b}/`. **Status: fixed** — the folder naming is now consistent (`qwen_3b`/`qwen_7b`, underscore, on both sides); it used to be hyphenated (`qwen-3b`/`qwen-7b`) under `no-few-shots/` only (see ISSUES §3).
2. **LoRA-level few-shot** (Qwen only, tangled with epoch count): `run_registry.json`'s `lora_1ep_fs` (few-shot) is the only few-shot LoRA run; all multi-epoch runs (`lora_2ep_nofs`, `lora_3ep_nofs`) are no-few-shot. So the LoRA-side "few-shot effect" is confounded with "epoch count" — there is no clean 2-epoch-with-few-shot run to isolate the two variables. The poster's framing ("few-shot prompting matters for terminology about as much as LoRA") is the honest way to describe this confound, not a clean ablation.
3. **The registry's `use_few_shot` flag for base runs is itself wrong.** `run_registry.json` sets `"use_few_shot": false` for both `Qwen2.5-3B/qwen_base` and `Qwen2.5-7B/qwen_base`, but the actual 7B base results (and `comparisons.xlsx`'s own column label `qwen_base_with_few_shots`) show 3 few-shot examples were in fact used for that run. `metrics_parser.py` only trusts the registry flag when a run's `metrics_summary.json` predates the `use_few_shot` field — which is exactly the 7B base run's case. Don't trust the registry's few-shot flag for base runs without cross-checking `comparisons.xlsx`.
4. `filter_test_sentence_overlap.py` and `filter_test_term_overlap.py` both default to output directories (`data/test_cleaned_gpt`, `data/test_cleaned`) that don't match the actually-committed directories (`test_cleaned_by_sentences/`, `test_cleaned_by_terms/`) — as committed, neither script can regenerate the current data from scratch without passing `--output-dir` explicitly.

**Decision needed from user:** report only the baseline-level few-shot comparison (clean, 3 models × with/without) and describe the LoRA-side entanglement as a limitation, or invest in running the missing `lora_2ep_fs`/`lora_3ep_fs` configs first? See ISSUES §6.

#### 3.4.2 Data-leakage honesty check ("good vs bad data") — 🔴 fully undocumented in any README
**Why this section exists:** dev_v2 is used as a training-set proxy for the shared task (see §3.3) because the real shared-task training data isn't available. But dev_v2 and dev_v1 are drawn from overlapping SAP documentation, so some dev_v2 lines are near-duplicates of dev_v1 test lines — fine-tuning on them would leak test signal and make LoRA's gains look bigger than they are.

**What was actually done** (reconstructed from [`experiments/04_lora_finetuning/scripts/remove_dev_v2_overlap.py`](../experiments/04_lora_finetuning/scripts/remove_dev_v2_overlap.py) and its (now-deleted, see ISSUES §1) report output):
1. dev_v2 started at **2000 lines/language** (verified: current `data/dev_v2/dev_v2_original/*.jsonl` is still 2000 lines/language).
2. **500 lines/language removed** where the English source *and* target already appeared verbatim in dev_v1 — exact-overlap deduplication, kept = 1500/language. (Recovered from git history, commit `5f9e31f`: `filter_unique_en: 500, input_lines: 2000, removed: 500, kept: 1500` for all three language pairs.)
3. **2–3 more lines/language removed** to reserve as few-shot prompt examples (commit `1fc187d`), leaving the training files at their current size — verified: `experiments/04_lora_finetuning/data/training/*.jsonl` = 1498 (ende) / 1497 (enes) / 1497 (enru) lines, i.e. 1500 minus 2–3.
4. Even after exact-overlap removal, some dev_v1 test sentences remain **near-duplicates** of dev_v2 training sentences (differing by only a few words) — this is the poster's stated caveat: *"some dev_v1 test sentences overlap lexically with dev_v2 training data."*
5. To check whether this near-duplicate leakage inflates LoRA's apparent gains, the test set was split into a **"good"** subset (no near-duplicate in training) and a **"bad"** subset (near-duplicate present), using [`experiments/04_lora_finetuning/scripts/filter_test_sentence_overlap.py`](../experiments/04_lora_finetuning/scripts/filter_test_sentence_overlap.py) (≥50% token containment ratio = "overlap"). Results for both subsets were evaluated for GPT and Qwen-7B-base, and manually compiled into `comparisons.xlsx`'s **"good vs bad data"** sheet.

**What the data shows** (from `comparisons.xlsx`, macro-avg BLEU): GPT bad=49.34 vs good=46.18; Qwen-base bad=38.84 vs good=38.07; Qwen-LoRA (7B, 2ep, no-few-shot) bad=56.28 vs good=46.97. All three models score *higher* on the "bad" (leakage-suspected) subset than the "good" subset, and LoRA's bad-vs-good gap (56.28 → 46.97, −9.3 BLEU) is larger than GPT's (49.34 → 46.18, −3.2) or Qwen-base's (38.84 → 38.07, −0.8) — i.e., **the data itself supports the "LoRA gains may be partly leakage-inflated" caveat**, and quantifies it: on the leakage-free "good" subset, LoRA's edge over GPT shrinks from +6.9 BLEU (bad subset) to +0.8 BLEU (good subset).

**Caveats to state plainly in the paper:**
- This check only used the `test_cleaned_by_sentences` split, and even that split is the *only* one worth reporting: the parallel `test_cleaned_by_terms` split (filtering on shared *term* overlap rather than whole-sentence overlap) turns out heavily skewed once inspected — only 11–14 "good" sentences vs. 483–486 "bad" sentences per language (almost every dev_v1 sentence shares a term-dictionary entry with dev_v2 training data), so it wouldn't give a statistically meaningful comparison even if run. It was prepared (`data/test_cleaned_by_terms/{data_good,data_bad}/`) but **never evaluated** — no matching `results/.../test_cleaned_by_terms/` exists anywhere. ⚫ Recommendation: don't run it; note the skew as the reason in the paper instead.
- Only 2 of the 5 comparison blocks in the "good vs bad data" sheet were produced by [`fill_good_vs_bad_gpt.py`](../experiments/04_lora_finetuning/scripts/fill_good_vs_bad_gpt.py) (GPT and Qwen-base blocks); the `qwen_lora` blocks (blocks 3–5) have no generating script anywhere in the repo and appear hand-entered — re-verify those numbers against `results/Qwen2.5-7B/qwen_lora_no_few_shots_2_epochs/test_cleaned_by_sentences/` before citing them.
- **The Qwen-base "good" column has a labeling bug**: in `fill_good_vs_bad_gpt.py`, the `qwen_base` block's `good_path` points at the plain, unfiltered `qwen_base/metrics_summary.json` (the full 500-line test set), not an actual good-filtered subset — so the Qwen-base row of this table is not really "good vs bad," it's "bad-filtered vs. full-unfiltered." Only the GPT row and the (unscripted, hand-entered) LoRA rows compare a true good-filtered subset against a true bad-filtered subset. Fix the script (point `good_path` at a real `qwen_base/test_cleaned_by_sentences/data_good/metrics_summary.json`, generating it first if it doesn't exist) before citing the Qwen-base line of this comparison.
- Only run for Qwen **7B**, not 3B.
- `data/dev_v2/dev_v2_original/training set/*.jsonl` (note the literal space in the folder name) and `experiments/04_lora_finetuning/data/training/*.jsonl` hold the same derived, post-removal training data checked in twice under two paths, with nothing in the repo indicating which is canonical — pick one before writing up exact training-set sizes.

## Report-table cross-reference

| Existing file | Produced by | Maps to paper table/figure |
| -------------- | ----------- | --------------------------- |
| `experiments/01_term_expansion_by_model/report/dev_v1_original_model_comparison.xlsx` (sheet `metrics`) | `experiments/01_term_expansion_by_model/scripts/compare_models_to_excel.py` | Model comparison table (§3.2) |
| `experiments/02_term_expansion_by_language_pair/report/dev_v1_original_language_comparison.xlsx` | `experiments/02_term_expansion_by_language_pair/scripts/compare_languages_to_excel.py` | Language comparison table (§3.1) |
| `experiments/03_dataset_comparison/report/dev_v1_original_vs_dev_v2_gpt_dataset_comparison.xlsx` | `experiments/03_dataset_comparison/scripts/compare_datasets_to_excel.py` | dev_v1 vs dev_v2 table (§3.3) |
| `experiments/03_dataset_comparison/report/dev_v1_original_vs_dev_v1_dictionary_gpt_comparison.xlsx` | `experiments/03_dataset_comparison/scripts/compare_v1_variants_to_excel.py` | Dictionary vs original table (§3.1) |
| `experiments/04_lora_finetuning/report/results_Qwen2.5-{3B,7B}.xlsx`, `results_GPT4o-mini.xlsx` | `experiments/04_lora_finetuning/scripts/export_finetuning_report.py` | LoRA main results, epoch ablation (§3.4) |
| `experiments/04_lora_finetuning/report/comparisons.xlsx` sheet `good vs bad data` | `experiments/04_lora_finetuning/scripts/fill_good_vs_bad_gpt.py` + manual edits | Leakage honesty-check table (§3.4.2) |
| `poster/figures/fig_term_expansion.pdf` | `experiments/01_term_expansion_by_model/scripts/figure_model_comparison.py` | Figure, §3.2 |
| `poster/figures/fig_expansion_strategies.pdf` | `experiments/02_term_expansion_by_language_pair/scripts/figure_mode_comparison.py` | Figure, §3.1 |
| `poster/figures/fig_dev_v1_vs_dev_v2_training.pdf` | `experiments/03_dataset_comparison/scripts/figure_dataset_comparison.py` | Figure, §3.3 |
| `poster/figures/fig_lora_finetuning.pdf` | `experiments/04_lora_finetuning/scripts/figure_lora_finetuning.py` | Figure, §3.4 |

## Open questions

See [`ISSUES.md`](ISSUES.md) §6 for full detail. Short version:
1. Report only the baseline-level few-shot comparison, or also fix the LoRA-side epoch/few-shot confound first (§3.4.1)?
2. Run the missing `test_cleaned_by_terms` evaluation before writing, or write around the gap and mention it as future work (§3.4.2)?
3. Re-verify (or re-generate via a script) the hand-entered `qwen_lora` rows in the "good vs bad data" sheet before citing them (§3.4.2)?
