# Cleanup notes for report writing

Every messy, undocumented, or broken thing found while planning [`README.md`](README.md), grouped by area. Each entry: what's wrong, where, why it matters for the paper, and a suggested fix or an explicit question. Nothing here is guessed — everything was verified by opening the actual file/script/commit named.

## 1. Lost/irrecoverable-in-current-tree artifacts

- **`remove_overlap_report.json` and the `dev_v2/deduped/` directory do not exist anywhere in the current `poster-correction` branch.** They only survive in git history (commit `5f9e31f`) and in unrelated old worktrees.
  - **Recovered numbers** (from `git show 5f9e31f:data/dev_v2/deduped/remove_overlap_report.json`): dev_v2 started at 2000 lines/language pair; exactly 500/language were removed for exact source+target overlap with dev_v1; 1500/language kept.
  - **Current-tree numbers**: `data/dev_v2/dev_v2_original/*.jsonl` = 2000 lines/language (this is the *pre*-removal file — the tree currently ships the un-deduplicated version as "original"). The post-removal, post-few-shot-trim training files (`experiments/05_lora_finetuning/data/training/*.jsonl`) = 1498 (ende) / 1497 (enes) / 1497 (enru), consistent with 1500 minus 2–3 lines reserved for few-shot examples (commit `1fc187d`).
  - **Status: fixed (script relocated + repaired).** The regenerating script moved to [`experiments/05_lora_finetuning/scripts/remove_dev_v2_overlap.py`](../experiments/05_lora_finetuning/scripts/remove_dev_v2_overlap.py) as part of the `src/analysis`/`src/data_preparation` cleanup, and its `term_utils.py`-derived path constants (`data/raw/`, `data/interim/`, `data/processed/`) were repointed at the current layout (`data/dev_v1/dev_v1_original/`, `data/dev_v2/dev_v2_original/`). It now writes to `experiments/05_lora_finetuning/data/dev_v2_deduped/` (a new directory, chosen so it doesn't silently overwrite the already-committed `data/training/`) instead of the old `data/processed/dev_v2/deduped/`. **Still needs:** actually running it once (requires `OPENROUTER_API_KEY`-free, pure filtering — no API cost) to regenerate `remove_overlap_report.json` and close the "silent gap" described above; not run as part of this code-move pass.

- **`data/dev_v2/dev_v2_original/training set/` duplicates `experiments/05_lora_finetuning/data/training/`** (literal space in the folder name; same line counts). Two copies of the same derived, post-removal dataset checked in under two different paths with nothing indicating which is canonical.
  - **Fix:** pick one as canonical, delete or symlink the other, and update whichever script/notebook still reads from the non-canonical path.

## 2. `05_lora_finetuning`'s README documents a pipeline that doesn't exist

`experiments/05_lora_finetuning/README.md` (lines ~33–92) describes an Excel export producing three workbooks — `results_Qwen2.5-3B.xlsx`, `results_Qwen2.5-7B.xlsx`, `results_GPT4o-mini.xlsx` — each with a specific documented sheet set (`main_results`, `base_vs_lora`, `epoch_ablation`, `experiment_config`, `training_loss_meta`, `training_loss`, etc.). **None of these three files exist anywhere in the repo.** The only workbook actually committed under `experiments/05_lora_finetuning/report/` is `comparisons.xlsx`, which has a completely different, partly-hand-filled sheet set (`Qwen2.5-3B`, `Qwen2.5-7B`, `qwen_base_lora_with_few_shots`, `qwen_lora_with_no_few_shots_x0009_`, `qwen_lora_no_few_shots_n_epoch_x0009_`, `best-models`, `good vs bad data`).

`scripts/run_registry.json`'s own `workbook` fields still point at the three missing files — the registry is stale, direct evidence of an abandoned/superseded export pipeline (`export_finetuning_report.py` + `sheet_builders.py` presumably built `comparisons.xlsx` at some point, or a different, undocumented process did).

- **Why it matters:** anyone (including the report writer) following the README to "regenerate the source tables" will run a script that either fails or produces files that were never actually used downstream. `comparisons.xlsx` is the real source of truth today but is undocumented as such.
- **Fix:** either (a) rewrite `05_lora_finetuning/README.md`'s Excel-export section to describe `comparisons.xlsx` and its actual sheets, or (b) actually run/fix `export_finetuning_report.py` to produce the three documented workbooks and make `comparisons.xlsx` the deprecated/legacy file. Pick one before the paper cites either.

## 3. Undocumented or partially-documented experiments

- **"Good vs bad data" leakage honesty check** (see `README.md` §3.4.2 for the full reconstruction) — not mentioned in any README anywhere in the repo. Only visible via `comparisons.xlsx`'s `good vs bad data` sheet and its partial generator script `fill_good_vs_bad_gpt.py`.
  - **Bug found:** `fill_good_vs_bad_gpt.py`'s Qwen-base block wires `good_path` to the plain, unfiltered `qwen_base/metrics_summary.json` — not an actual good-filtered subset. So the Qwen-base row of the sheet is silently "bad-filtered vs. full-unfiltered," not "good vs bad" like the GPT row and the (separately, manually entered) LoRA rows. **This makes the Qwen-base row in the current sheet directly misleading if cited as-is.**
  - Only 2 of 5 comparison blocks (GPT, Qwen-base) are script-generated; the 3 `qwen_lora` blocks have no generating script anywhere and appear hand-entered.
  - Only evaluated for Qwen **7B**; no 3B good/bad results exist.
  - `filter_test_sentence_overlap.py` / `filter_test_term_overlap.py` both default to output directories (`data/test_cleaned_gpt`, `data/test_cleaned`) that don't match the actually-committed directories (`test_cleaned_by_sentences/`, `test_cleaned_by_terms/`) — as committed, neither script reproduces the current data without manually passing `--output-dir`.
  - `filter_test_term_overlap.py` overwrites its input file in place (per subagent investigation) — re-running it destructively modifies source data rather than writing a fresh copy, unlike its sentence-level sibling.

- **`test_cleaned_by_terms` split prepared but never evaluated**, and turns out heavily skewed once inspected: 11–14 "good" sentences vs. 483–486 "bad" sentences per language (nearly every dev_v1 sentence shares a term-dictionary entry with dev_v2 training data). No `results/.../test_cleaned_by_terms/` directory exists anywhere.
  - **Recommendation:** don't run it — note the skew as the reason it wasn't used, rather than silently omitting it.

- **Few-shot is split across two independently-produced, inconsistently-named result trees**, unified only inside `experiments/05_lora_finetuning/scripts/figure_exp5.py`:
  1. Baseline-level: `results/dev_v1/original/{no-few-shots,with-few-shots}/` (repo root) — GPT + Qwen 3B/7B, no fine-tuning. Sibling folders even disagree on naming: `no-few-shots/` uses `qwen-3b`/`qwen-7b` (hyphen), `with-few-shots/` uses `qwen_3b`/`qwen_7b` (underscore), and the two also differ in internal directory shape.
  2. LoRA-level: `run_registry.json`'s `lora_1ep_fs` is the only few-shot LoRA run; all multi-epoch runs are no-few-shot — so on the LoRA side, "few-shot effect" and "epoch count" are confounded. There is no `lora_2ep_fs`/`lora_3ep_fs` to isolate them.
  3. **`run_registry.json`'s `use_few_shot: false` for both Qwen base runs is itself wrong** — the actual 7B base run used 3 few-shot examples per language (confirmed by `comparisons.xlsx`'s own column header `qwen_base_with_few_shots`, and by the notebook's `SAMPLE_SENTENCES`/`USE_FEW_SHOT` logic). `metrics_parser.py` only falls back to the registry flag when a run's `metrics_summary.json` predates the `use_few_shot` field — which is exactly the base runs' case, so the wrong flag silently propagates into any table built from the registry.
  - **Fix:** correct the registry's `use_few_shot` flags for both base runs; rename the `no-few-shots`/`with-few-shots` subfolders to consistent casing; decide (see §6) whether to invest in the missing `lora_2ep_fs`/`lora_3ep_fs` runs or explicitly describe the confound as a limitation.

## 4. Excel workbook mess

- **`experiments/01_term_expansion_by_model/report/dev_v1_original_model_comparison.xlsx` has 3 sheets** (`metrics`, `Sheet1`, `terms expansion types`) where its 3 sibling files (`dev_v1_cleaned_...`, `dev_v1_expand_...`, `dev_v2_...`) each have exactly 1 sheet (`metrics`). `Sheet1` is openpyxl's un-renamed default sheet name, containing a near-duplicate table (headed `data` instead of `mode`, extra Term-Accuracy columns) — leftover scratch output. `terms expansion types` is a third, differently-shaped table. Neither extra sheet is mentioned in the README.
  - **Fix:** delete `Sheet1` and `terms expansion types` from this one file (or fold their unique content into `metrics` if anything in them isn't already redundant), so all 4 sibling files have the same, single-sheet shape. Cite only the `metrics` sheet until then.

- **`experiments/02_term_expansion_by_language_pair/report/dev_v1_original_mode_comparison.xlsx` has a different internal layout than its 3 siblings.** All 4 files have exactly 1 sheet named `modes`, but the 3 siblings (`dev_v1_cleaned_...`, `dev_v1_expand_...`, `dev_v2_...`) are each an 11-row wide table, while `dev_v1_original_...` is 51 rows: a title row `'Original data'` followed by four stacked long-format blocks (Original / Expand / Cleaned / devset_2) — effectively cramming what the other three files keep as separate per-variant files into one sheet, in a different shape.
  - **Fix:** decide whether `dev_v1_original_mode_comparison.xlsx` should be reshaped to match its siblings (1 wide table) or whether the siblings are the ones that should be consolidated to match it — right now the four "identically named" files are not structurally comparable, which will bite whoever writes the table-generation code for the paper.

- **`_x0009_` sheet-name corruption in `comparisons.xlsx`.** `qwen_lora_with_no_few_shots_x0009_` and `qwen_lora_no_few_shots_n_epoch_x0009_` contain a literal `_x0009_`, openpyxl's escape for an embedded raw TAB character — someone hit Tab while renaming an Excel sheet tab and it got baked into the name.
  - **Fix:** rename both sheets in Excel/openpyxl to clean names (e.g. `qwen_lora_no_few_shots`, `qwen_lora_no_few_shots_epoch_ablation`) before the file is used as a citation source.

- **Mixed autogenerated/hand-edited sheets with no record of which cells are which.** `best-models` and `good vs bad data` in `comparisons.xlsx` mix script-written cells (GPT, Qwen-base blocks) with hand-entered ones (LoRA blocks) with no marker distinguishing them — see §3.
  - **Fix going forward:** either script-generate every cell (write the missing `qwen_lora` block generator) or clearly flag hand-edited sheets/cells (e.g. a header note or a distinct fill color) so a reader — or the report writer six months later — can tell which numbers are traceable to a script run and which aren't.

## 5. Stale README references (already known in-repo, restated here for completeness)

- Root `README.md` describes a `data/raw|interim|processed` layout; the actual tree is `data/dev_v1|dev_v2|shared_task`. Also references `experiments/04_baseline/`; the actual folder is `experiments/baseline/` (no number prefix). Both already implicitly acknowledged by the repo (e.g. `01_term_expansion_by_model/README.md` cites `04_baseline` too, so this propagated).
- `data/dev_v1/dev_v1_dictionary/apply_report.json`'s own `input`/`output` path fields are stale, referencing old `data/dev_v1/original/`, `data/dev_v1/dictionary/` naming rather than the real `dev_v1_original`/`dev_v1_dictionary` folder names — a self-inconsistency inside a single generated report file.
- `data/dev_v1/dev_v1_dictionary/{ende,enes,enru}_ambiguous_skipped.jsonl` — orphaned diagnostic output from the dictionary-application script (37/46/16 ambiguous terms skipped per language, per `apply_report.json`), not referenced or consumed anywhere downstream. Fine to leave as a diagnostic artifact, but worth a one-line mention in `03_dataset_comparison/README.md` (or wherever the dictionary experiment ends up in the paper) so it isn't mistaken for dead weight.
- `poster/figures/fig_exp4_dictionary_vs_original.pdf`/`.png` — orphaned poster artifacts; `03_dataset_comparison/README.md` already self-documents this as unreproducible from any current script. Don't cite this figure as live.

## 6. Open questions for the user

1. **Few-shot scope for the paper**: report only the clean baseline-level few-shot comparison (GPT/Qwen 3B/7B, no fine-tuning) and describe the LoRA-side epoch/few-shot confound as a stated limitation — or invest time in running the missing `lora_2ep_fs`/`lora_3ep_fs` configs first for a cleaner ablation?
2. **`test_cleaned_by_terms`**: skip it and explain the class-imbalance reason in the paper (recommended, given the 11–14 vs. 483–486 skew), or is there a reason to still want it run?
3. **Good-vs-bad-data table**: fix `fill_good_vs_bad_gpt.py`'s Qwen-base labeling bug and regenerate a real good-filtered Qwen-base subset before citing that row — agree, or is the current "bad vs full" comparison actually what was intended?
4. **LoRA export pipeline** (`comparisons.xlsx` vs. the README's documented-but-missing 3-workbook export): rewrite the README to match `comparisons.xlsx` as it exists, or fix `export_finetuning_report.py` to actually produce the documented workbooks and treat `comparisons.xlsx` as legacy?
5. **Canonical training-set path**: `data/dev_v2/dev_v2_original/training set/` vs. `experiments/05_lora_finetuning/data/training/` — which one is canonical, so the other can be removed/symlinked?
