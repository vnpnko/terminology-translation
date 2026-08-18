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

### 3.4 LoRA fine-tuning (Qwen2.5-3B / 7B, 1–3 epochs)
- **RQ:** Does LoRA fine-tuning on dev_v2 close the gap between open Qwen models and GPT-4o-mini on terminology-constrained translation?
- **Repo location:** [`experiments/04_lora_finetuning/`](../experiments/04_lora_finetuning/README.md), evaluated on `proper_term` mode only.
- **Canonical runs** (from [`scripts/run_registry.json`](../experiments/04_lora_finetuning/scripts/run_registry.json)): `base_few_shot` (no fine-tuning, few-shot), `lora_1_epoch_few_shot`, `lora_1_epoch_zero_shot`, `lora_2_epoch_zero_shot`, `lora_3_epoch_zero_shot`, for both 3B and 7B, plus `gpt_base`. Run ids use the standard `zero_shot`/`few_shot` vocabulary (see experiment README § Naming standard) — the underlying result folders keep their original names.
- **Tables:** five comparison workbooks in `experiments/04_lora_finetuning/report/` (see [experiment README § Report tables](../experiments/04_lora_finetuning/README.md#report-tables)): `leakage_honesty_check.xlsx`, `base_few_shot_vs_lora_zero_shot_1_epoch.xlsx`, `best_models.xlsx`, `epoch_ablation.xlsx`, `zero_shot_vs_few_shot_ablation.xlsx`. These replaced the old hand-assembled `comparisons.xlsx`. All 5 are now fully script-generated (`compare_leakage_honesty_check_to_excel.py`/`compare_epochs_to_excel.py`/`compare_few_shots_to_excel.py`/`compare_base_vs_lora_to_excel.py`/`compare_best_models_to_excel.py`) — no hand-assembled workbooks remain. `run_registry.json`'s own `workbook` field still points at the never-produced `results_Qwen2.5-{3B,7B}.xlsx`/`results_GPT4o-mini.xlsx` legacy pipeline, which is dead and unused (see ISSUES §2). Note: `best_models.xlsx` now cites the **2-epoch** LoRA run as "best" (previously mislabeled as 3-epoch) — see the experiment README's note on the overfitting evidence.
- **Figure:** `poster/figures/fig_lora_finetuning.pdf` — epoch ablation vs. GPT-4o-mini.
- **Poster findings to reuse:** LoRA gains build through epochs, peak ~2 epochs; 7B term accuracy stable, 3B term accuracy drops without few-shot but recovers after fine-tuning; best 7B LoRA closes much of the BLEU gap to GPT, especially EN→RU, while GPT few-shot stays strongest on term accuracy.

#### 3.4.1 Few-shot ablation — 🟡 two separate, differently-scoped experiments exist under this name
There are **two distinct few-shot comparisons** in the repo; the report needs to pick which one(s) it reports and name them so readers don't conflate them:

1. **Baseline-level few-shot** (GPT + Qwen 3B/7B, no fine-tuning): `results/dev_v1/original/zero_shot/{gpt,qwen_3b,qwen_7b}/` vs `results/dev_v1/original/few_shot/{gpt,qwen_3b,qwen_7b}/`. **Status: fixed** — the folder naming is now consistent (`qwen_3b`/`qwen_7b`, underscore, on both sides), and the top-level folders were renamed from the hyphenated `no-few-shots`/`with-few-shots` to `zero_shot`/`few_shot` to match the repo-wide naming standard (see ISSUES §3).
2. **LoRA-level few-shot** (Qwen only, tangled with epoch count): `run_registry.json`'s `lora_1_epoch_few_shot` is the only few-shot LoRA run; all multi-epoch runs (`lora_2_epoch_zero_shot`, `lora_3_epoch_zero_shot`) are zero-shot. So the LoRA-side "few-shot effect" is confounded with "epoch count" — there is no clean 2-epoch-few-shot run to isolate the two variables. The poster's framing ("few-shot prompting matters for terminology about as much as LoRA") is the honest way to describe this confound, not a clean ablation.
3. ~~**The registry's `use_few_shot` flag for base runs is itself wrong.**~~ — ✅ fixed: `run_registry.json` used to set `"use_few_shot": false` for both `Qwen2.5-3B/qwen_base` and `Qwen2.5-7B/qwen_base`, but the actual base results show 3 few-shot examples were in fact used for that run. The flag is now `true` and the run id renamed `base_few_shot` to make it self-documenting (see experiment README § Naming standard).
4. `filter_test_sentence_overlap.py` and `filter_test_term_overlap.py` both default to output directories (`data/test_cleaned_gpt`, `data/test_cleaned`) that don't match the actually-committed directories (`test_cleaned_by_sentences/`, `test_cleaned_by_terms/`) — as committed, neither script can regenerate the current data from scratch without passing `--output-dir` explicitly.

**Decision needed from user:** report only the baseline-level few-shot comparison (clean, 3 models × zero/few-shot) and describe the LoRA-side entanglement as a limitation, or invest in running the missing `lora_2_epoch_few_shot`/`lora_3_epoch_few_shot` configs first? See ISSUES §6.

#### 3.4.2 Data-leakage honesty check (`leakage_honesty_check.xlsx`) — 🔴 fully undocumented in any README
**Why this section exists:** dev_v2 is used as a training-set proxy for the shared task (see §3.3) because the real shared-task training data isn't available. But dev_v2 and dev_v1 are drawn from overlapping SAP documentation, so some dev_v2 lines are near-duplicates of dev_v1 test lines — fine-tuning on them would leak test signal and make LoRA's gains look bigger than they are.

**What was actually done** (reconstructed from [`experiments/04_lora_finetuning/scripts/remove_dev_v2_overlap.py`](../experiments/04_lora_finetuning/scripts/remove_dev_v2_overlap.py) and its (now-deleted, see ISSUES §1) report output):
1. dev_v2 started at **2000 lines/language** (verified: current `data/dev_v2/dev_v2_original/*.jsonl` is still 2000 lines/language).
2. **500 lines/language removed** where the English source *and* target already appeared verbatim in dev_v1 — exact-overlap deduplication, kept = 1500/language. (Recovered from git history, commit `5f9e31f`: `filter_unique_en: 500, input_lines: 2000, removed: 500, kept: 1500` for all three language pairs.)
3. **2–3 more lines/language removed** to reserve as few-shot prompt examples (commit `1fc187d`), leaving the training files at their current size — verified: `experiments/04_lora_finetuning/data/training/*.jsonl` = 1498 (ende) / 1497 (enes) / 1497 (enru) lines, i.e. 1500 minus 2–3.
4. Even after exact-overlap removal, some dev_v1 test sentences remain **near-duplicates** of dev_v2 training sentences (differing by only a few words) — this is the poster's stated caveat: *"some dev_v1 test sentences overlap lexically with dev_v2 training data."*
5. To check whether this near-duplicate leakage inflates LoRA's apparent gains, the test set was split into a **"good"** subset (no near-duplicate in training) and a **"bad"** subset (near-duplicate present), using [`experiments/04_lora_finetuning/scripts/filter_test_sentence_overlap.py`](../experiments/04_lora_finetuning/scripts/filter_test_sentence_overlap.py) (≥50% token containment ratio = "overlap"). Results for both subsets were evaluated for GPT and Qwen-7B (base and LoRA), and compiled into **`leakage_honesty_check.xlsx`** (single sheet `overlap_vs_no_overlap_data`, via [`compare_leakage_honesty_check_to_excel.py`](../experiments/04_lora_finetuning/scripts/compare_leakage_honesty_check_to_excel.py); column groups `overlap_data`/`no_overlap_data`, named for the split criterion rather than a "bad"/"good" judgment), 3 stacked model blocks in this order: `qwen_base` (untrained, control), `gpt` (closed model, never exposed to `dev_v2` training data, control), `qwen_lora` (trained, the model under test) — controls first so the overlap-vs-no-overlap gap visibly grows with training exposure.

**What the data shows** (from `leakage_honesty_check.xlsx`, macro-avg BLEU — not itself a column in the sheet, computed from the per-language rows): qwen_base (untrained) overlap=38.84 vs no-overlap=38.07 (gap −0.8); GPT (closed, control) overlap=49.34 vs no-overlap=46.18 (gap −3.2); qwen_lora (trained) overlap=56.28 vs no-overlap=46.97 (gap −9.3). All three models score *higher* on the `overlap_data` (leakage-suspected) subset than `no_overlap_data`, and the gap grows with training exposure — untrained qwen_base's gap is negligible, GPT's is a bit larger (inherent sentence difficulty, since GPT never saw `dev_v2`), and the trained qwen_lora's is ~11x qwen_base's — read as leakage-inflation evidence on top of the difficulty confound, not instead of it. i.e., **the data supports a "LoRA gains may be partly leakage-inflated, on top of a real difficulty confound" caveat**, and quantifies it: on the leakage-free `no_overlap_data` subset, LoRA's edge over GPT shrinks from +6.9 BLEU (`overlap_data`) to +0.8 BLEU (`no_overlap_data`).

**Caveats to state plainly in the paper:**
- **`overlap_data`/`no_overlap_data` are confounded with sentence length and terminology density, not just training-set overlap.** Directly measured from `data/test_cleaned_by_sentences/{data_bad,data_good}/{ende,enes,enru}_dev_v1_test.jsonl` (164 sentences/subset/language): `overlap_data` sentences are **30–49% shorter** on average (e.g. `enes`: 8.6 vs 17.0 words/sentence; `ende`: 9.5 vs 13.5; `enru`: 8.3 vs 13.3), while per-sentence term counts are nearly flat between subsets (~2.1–2.5 terms/sentence both ways, confirmed via `proper_terms`+`random_terms` counts) — so `overlap_data` has meaningfully higher terminology *density* (terms per word), most pronounced on `enes` (38.8% vs 17.8%, more than double). Total unique terms per subset are also nearly identical (e.g. `ende`: 179 vs 191, from `metrics_summary.json`'s `terminology_accuracy.total_terms`), ruling out vocabulary differences as the cause. **Implication:** short, dense sentences are inherently easier to translate for *any* model, not just ones exposed to `dev_v2` in training — this is exactly why even GPT (never trained on `dev_v2`) shows a real `overlap_data`-vs-`no_overlap_data` gap (§ "What the data shows" above, gap ≈−3.2 BLEU). The leakage-inflation reading for `qwen_lora` (gap ≈−9.3) should be stated as "on top of this difficulty confound," not as if the confound were absent.
- This check only used the `test_cleaned_by_sentences` split, and even that split is the *only* one worth reporting: the parallel `test_cleaned_by_terms` split (filtering on shared *term* overlap rather than whole-sentence overlap) turns out heavily skewed once inspected — only 11–14 "good" sentences vs. 483–486 "bad" sentences per language (almost every dev_v1 sentence shares a term-dictionary entry with dev_v2 training data), so it wouldn't give a statistically meaningful comparison even if run. It was prepared (`data/test_cleaned_by_terms/{data_good,data_bad}/`) but **never evaluated** — no matching `results/.../test_cleaned_by_terms/` exists anywhere. ⚫ Recommendation: don't run it; note the skew as the reason in the paper instead.
- ✅ Resolved: `leakage_honesty_check.xlsx` is now fully generated by `compare_leakage_honesty_check_to_excel.py` — every cell (GPT, `qwen_base`, `qwen_lora`) is derived directly from `metrics_summary.json`, replacing the old file's hand-entered `qwen_lora` blocks and the buggy `fill_good_vs_bad_gpt.py` (now removed). The `qwen_base` "good" labeling bug is also resolved: `qwen_base/test_cleaned_by_sentences/data_good/metrics_summary.json` exists and was verified to be a real good-filtered subset (its numbers already matched the old file exactly), so the earlier concern about it pointing at unfiltered data no longer applies.
- Only run for Qwen **7B**, not 3B.
- `data/dev_v2/dev_v2_original/training set/*.jsonl` (note the literal space in the folder name) and `experiments/04_lora_finetuning/data/training/*.jsonl` hold the same derived, post-removal training data checked in twice under two paths, with nothing in the repo indicating which is canonical — pick one before writing up exact training-set sizes.

## Report-table cross-reference

| Existing file | Produced by | Maps to paper table/figure |
| -------------- | ----------- | --------------------------- |
| `experiments/01_term_expansion_by_model/report/dev_v1_original_model_comparison.xlsx` (sheet `metrics`) | `experiments/01_term_expansion_by_model/scripts/compare_models_to_excel.py` | Model comparison table (§3.2) |
| `experiments/02_term_expansion_by_language_pair/report/dev_v1_original_language_comparison.xlsx` | `experiments/02_term_expansion_by_language_pair/scripts/compare_languages_to_excel.py` | Language comparison table (§3.1) |
| `experiments/03_dataset_comparison/report/dev_v1_original_vs_dev_v2_gpt_dataset_comparison.xlsx` | `experiments/03_dataset_comparison/scripts/compare_datasets_to_excel.py` | dev_v1 vs dev_v2 table (§3.3) |
| `experiments/03_dataset_comparison/report/dev_v1_original_vs_dev_v1_dictionary_gpt_comparison.xlsx` | `experiments/03_dataset_comparison/scripts/compare_v1_variants_to_excel.py` | Dictionary vs original table (§3.1) |
| `experiments/04_lora_finetuning/report/base_few_shot_vs_lora_zero_shot_1_epoch.xlsx` | `experiments/04_lora_finetuning/scripts/compare_base_vs_lora_to_excel.py` | Base-few-shot vs LoRA-1-epoch table (§3.4) |
| `experiments/04_lora_finetuning/report/best_models.xlsx` | `experiments/04_lora_finetuning/scripts/compare_best_models_to_excel.py` | Best LoRA vs GPT table (§3.4) |
| `experiments/04_lora_finetuning/report/epoch_ablation.xlsx` | `experiments/04_lora_finetuning/scripts/compare_epochs_to_excel.py` | Epoch ablation table (§3.4) |
| `experiments/04_lora_finetuning/report/zero_shot_vs_few_shot_ablation.xlsx` | `experiments/04_lora_finetuning/scripts/compare_few_shots_to_excel.py` | Few-shot ablation table (§3.4.1) |
| `experiments/04_lora_finetuning/report/leakage_honesty_check.xlsx` | `experiments/04_lora_finetuning/scripts/compare_leakage_honesty_check_to_excel.py` | Leakage honesty-check table (§3.4.2) |
| `poster/figures/fig_term_expansion.pdf` | `experiments/01_term_expansion_by_model/scripts/figure_model_comparison.py` | Figure, §3.2 |
| `poster/figures/fig_expansion_strategies.pdf` | `experiments/02_term_expansion_by_language_pair/scripts/figure_mode_comparison.py` | Figure, §3.1 |
| `poster/figures/fig_dev_v1_vs_dev_v2_training.pdf` | `experiments/03_dataset_comparison/scripts/figure_dataset_comparison.py` | Figure, §3.3 |
| `poster/figures/fig_lora_finetuning.pdf` | `experiments/04_lora_finetuning/scripts/figure_lora_finetuning.py` | Figure, §3.4 |

## Open questions

See [`ISSUES.md`](ISSUES.md) §6 for full detail. Short version:
1. Report only the baseline-level few-shot comparison, or also fix the LoRA-side epoch/few-shot confound first (§3.4.1)?
2. Run the missing `test_cleaned_by_terms` evaluation before writing, or write around the gap and mention it as future work (§3.4.2)?
