# Report writing plan

Companion to [`README.md`](README.md) (Experiments-section source of truth) and [`ISSUES.md`](ISSUES.md) (open gaps). This doc plans the **full ACL-style paper**, to be drafted section-by-section directly into the Overleaf project (`acl_latex.tex`, ACL conference template), reviewed one section at a time rather than generated in one shot.

## Confirmed scope decisions

- **Full paper**, all standard ACL sections — not just the Experiments section.
- **Excludes** `experiments/term_expansion/gpt_proposed_term_pipeline/` — flagged unreviewed in `ISSUES.md`; forward-referenced only as future work in the Conclusion, no cited numbers.
- **Does not reference** the earlier mock PDF (`acl_latex.pdf`) for any content, structure, or phrasing — built independently from this repo's own docs.
- **Target length**: ACL long paper, ~8 pages + references + appendix.
- **Authors**: Ivan Pylypenko, Robin Schukrafft, Ilnur Gayetov — TUM, supervised by Dr. Marion Di Marco (matches root `README.md`'s Authors section).
- **Backbone**: `report/README.md`'s §3.1–§3.4.2 Experiments outline is the source of truth for what to write about and in what order; content gets re-verified against current `.xlsx` output before being turned into prose, not copied blind.
- **Figures**: all-new, column-width, one per Experiments subsection max — poster figures are too dense/large to shrink legibly. Built by extending `shared/lib/analysis/plot_style.py` / `grouped_bar_figure_common.py`, not new plotting code from scratch.

## Section-by-section outline

### Title
"Prompting vs. LoRA Fine-Tuning for Terminology-Constrained Translation of SAP Software Documentation" (working title — independently descriptive, not drawn from the mock PDF).

### Abstract (~150–200 words)
- Problem: enterprise MT of SAP documentation, terminology mistranslation breaks UI consistency.
- Approach: prompt-based term-list conditioning (3 modes, 4 term-list variants) vs. LoRA fine-tuning of Qwen2.5, evaluated against GPT-4o-mini.
- States all 3 RQs and their short answers.
- Headline numbers — filled in once §5 prose is finalized and verified.

### 1. Introduction
- Opens with the problem framing already established in root `README.md`'s motivation paragraph (a validated repo doc — not the mock PDF).
- States the 3 RQs (mirroring `report/README.md`'s per-section RQs): does terminology conditioning help; does term-list construction method matter; can LoRA close the gap to GPT-4o-mini.
- Contributions as 3 bullets, one per RQ.
- No "paper roadmap" sentence (space-saving).

### 2. Related Work
- ~2 short paragraphs: (a) terminology-constrained MT (glossary injection, constrained decoding), citing the WMT25 Terminology Shared Task as the closest prior setup; (b) LoRA/PEFT adaptation, citing Hu et al. 2022.
- **TODO before compiling**: verify exact BibTeX for WMT25 shared task papers, Hu et al. 2022 (LoRA), Papineni et al. 2002 (BLEU), Popović 2015 (chrF), Qwen2.5 technical report, and the SAP `term_postedits` dataset. Do not fabricate entries — pull from ACL Anthology / arXiv / the dataset's own citation info.

### 3. Data and Task
- 3.1 Corpora — `dev_v1`/`dev_v2` definitions verbatim from `report/README.md`'s Canonical vocabulary table; dev_v2's WMT25-proxy caveat stated explicitly; one-line mention that `shared_task/` materials exist but aren't the training source (unavailable).
- 3.2 Record format — the JSONL example (source/reference/proper_terms/random_terms), as in root `README.md`.
- 3.3 Metrics — BLEU, chrF, term accuracy, macro/weighted consistency, defined per `report/README.md`'s Canonical vocabulary table; note macro-avg-over-language-pairs convention for prose, full per-language numbers deferred to Appendix.

### 4. Methods
- 4.1 Prompt conditioning — `no_term`/`proper_term`/`random_term`, few-shot convention (3 examples/lang unless noted).
- 4.2 Term-list variants — `original`/`expand`/`cleaned`/`dictionary`, one sentence each on construction method (sourced from each `experiments/term_expansion/*/README.md`).
- 4.3 Models and fine-tuning — GPT-4o-mini, Qwen2.5-3B/7B-Instruct; LoRA setup (epoch sweep 1–3, zero-shot/few-shot at inference, training set size 1497–1498 lines/lang post-dedup).
- One new compact pipeline-overview figure (built during drafting, not reused from the mock PDF).

### 5. Experiments
Fixed template per subsection: RQ → setup → table/figure → 2–4 sentence findings (numbers re-verified against current `.xlsx` output, not copied from any prior draft).

- **5.1** Terminology modes across language pairs — RQ from `report/README.md` §3.1. Table: `language_comparison.xlsx` (macro), figure: new column-width version of `fig_term_expansion_across_languages_*`.
- **5.2** Terminology modes across models — RQ from §3.2. Table: `model_comparison.xlsx`, figure: new version of `fig_term_expansion_across_models`.
- **5.3** dev_v1 vs. dev_v2 comparability — background-context-sized (not a full RQ), per §3.3's own framing; sets up 5.4.2.
- **5.4** LoRA fine-tuning — RQ from §3.4. Epoch ablation + best-model tables/figures; the "2-epoch best" claim explicitly carries its overfitting caveat (flagged for Limitations too).
  - **5.4.1** Few-shot ablation — both the baseline-level and LoRA-level comparisons, named separately per §3.4.1's own note that they're two distinct comparisons.
  - **5.4.2** Data-leakage honesty check — full subsection, condensed from the existing detailed write-up in `report/README.md` §3.4.2 (overlap/no-overlap split, difficulty-confound caveat, quantified leakage-inflation estimate).
- One qualitative wrong-term example table (like a "Table 1"), freshly pulled from real current prediction files in whichever subsection best illustrates it (likely 5.1 or 5.2) — not reused from the mock PDF's examples.

### 6. Discussion
- Written only after §5 prose is finalized. Synthesizes: (a) generic (BLEU/chrF) vs. terminology metrics disagreeing on "better"; (b) prompting vs. fine-tuning solving different parts of the problem; (c) the leakage caveat as a cross-cutting point, not just buried in 5.4.2.

### 7. Limitations (mandatory)
Independently written from `report/ISSUES.md` + `report/README.md`'s existing caveats (not the mock PDF's Limitations text):
- Narrow domain/language coverage (3 pairs, 1 domain).
- Modest eval set sizes (500/lang dev_v1, ~1500/lang dev_v2).
- Train/test lexical overlap (§3.4.2's difficulty-confound-adjusted leakage estimate).
- Oracle-term-list assumption — most prompt experiments assume a given/offline term list, not inference-time extraction (this is also where `gpt_proposed_term_pipeline` gets its forward-reference in the Conclusion, not here).
- Term accuracy is surface-form presence only, no morphology/agreement check.
- LoRA epoch-count "best" choice has overfitting evidence behind it (carried from §5.4).

### 8. Ethics / Broader Impact
2–3 sentences: public dataset (`SAP/software-documentation-data-set-for-machine-translation`), no PII, used under a course-project research context.

### 9. Conclusion
Restates the 3 RQ answers concisely; one sentence of future work pointing at dropping the oracle-term-list assumption — this is where `gpt_proposed_term_pipeline` gets acknowledged as ongoing/future work, without citing any of its unreviewed numbers.

### Acknowledgments
TUM course project, supervised by Dr. Marion Di Marco — matches root `README.md`.

### Author Contributions
Matches root `README.md`'s Authors section content (Ivan: Qwen/LoRA study + GPT baselines + poster assets; Robin: GPT-term pipeline + dictionary + figures; Ilnur: data-prep pipeline + restructure + shared eval refactor).

### References
BibTeX per the Related Work TODO above — verify, don't fabricate.

### Appendix
Full per-language tables for every experiment (mirrors `report/README.md`'s Report-table cross-reference list exactly — one appendix table per existing `.xlsx` file).

## Cross-cutting rules
- **Vocabulary lock**: use `report/README.md`'s Canonical vocabulary table terms verbatim everywhere (`no_term`/`proper_term`/`random_term`, `dev_v1`/`dev_v2`, `gpt-4o-mini`/`Qwen2.5-3B-Instruct`/`Qwen2.5-7B-Instruct`) — no re-terming, no shorthand drift.
- **Tables**: plain black-and-white, best value per column bolded — not the Excel report's Green/Red/Yellow fills (print-friendly ACL convention).
- **Process**: draft one section into `acl_latex.tex` at a time, review before moving to the next — this is the direct fix for the "first-shot" risk that prompted this plan.

## Open TODOs before this can go to prose
1. Verify all BibTeX entries (Related Work + References).
2. Re-verify every findings bullet against current `.xlsx` output before writing Experiments prose (no numbers carried over from any prior draft without a fresh check).
3. Design and build the new pipeline-overview figure (§4) and the new column-width result figures (§5) using the existing shared plotting scripts.
4. Pull fresh qualitative wrong-term examples from real current prediction files for the one example table.
