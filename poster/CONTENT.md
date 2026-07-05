# Poster Content

---

## Header

**Title:** Terminology-Constrained Machine Translation for SAP Documentation

**Subtitle:** EN→{DE, RU, ES} with prompt conditioning, GPT pipeline, and LoRA fine-tuning

**Authors:** Ivan Pylypenko, Robin Schukrafft, Ilnur Gayetov, Marion Di Marco

**Affiliation:** Technical University of Munich

---

## Column 1 — Motivation & Task

### Problem

- Enterprise MT must preserve SAP/IT domain terms (UI labels, product names, technical concepts).
- Generic MT often paraphrases or mistranslates terminology, producing inconsistent product documentation.

### Setup

- Independent TUM course project building on the setup of the **WMT25 Terminology Shared Task** (sentence-level track, terminology provided as constraints).

### Research Questions

- Does explicit terminology in the prompt improve BLEU, chrF, term accuracy and consistency?
- Does _expanding_ the term list (GPT vs external dictionary) help, and does cleaning it help more?
- Does LoRA fine-tuning on SAP data close the gap for open models (Qwen)?

### Data & Task

- **dev_v1:** 500 sentences per language (course evaluation set).
- **dev_v2:** ~1,500 sentences per language from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus.
- Each JSONL record: English source, target reference, `proper_terms` (domain IT term pairs), `random_terms` (control).
- Language pairs: EN→DE, EN→RU, EN→ES.

**Figure:** `figures/data_format.png` — example JSONL record with source, reference, proper/random terms.

### Metrics

- **BLEU / chrF** — general similarity to the reference (n-gram / character-level); _insensitive to terminology._
- **Term Accuracy %** — share of required proper terms whose correct target form appears in the output.
- **Macro Consistency** — mean consistency of each term's translation across the corpus (unweighted).
- **Weighted Consistency** — same, weighted by term frequency.

---

## Column 2 — Methods

### Prompt-Conditioning Modes

| Mode          | Description                     |
| ------------- | ------------------------------- |
| `no_term`     | Translate without terminology   |
| `proper_term` | Inject oracle domain term pairs |
| `random_term` | Inject non-domain control pairs |

### Term-Expansion Strategies (dev_v1, GPT-4o-mini)

| Strategy       | How terms are added                                                     |
| -------------- | ----------------------------------------------------------------------- |
| `original`     | Only the proper terms shipped with dev_v1                               |
| `GPT-expanded` | GPT extracts extra proper terms _in-context_ from each source sentence  |
| `dictionary`   | Terms mined from dev_v2 and matched back onto dev_v1 sources            |
| `GPT-cleaned`  | GPT filters the expanded list to keep only highly domain-specific terms |

### Models

- **GPT-4o-mini** — primary baseline
- **Qwen2.5-3B / 7B-Instruct** — open-model baselines

### LoRA Fine-Tuning

- Train on dev_v2 (1,500/lang); test on held-out dev_v1 (497/lang); 3 sentences reserved as few-shot examples.
- Qwen 3B/7B: base, LoRA 1–3 epochs, with/without few-shot.
- Best config: **7B LoRA, 2 epochs, no few-shot**.
- 7B LoRA 3 epochs begins to overfit: BLEU/chrF plateau while term accuracy inches up.

---

## Column 3 — Results

### 1. Terminology injection & expansion (GPT-4o-mini, `proper_term`, macro-avg EN→DE/RU/ES)

| Setting        | BLEU | chrF | Term Acc % | Macro Cons | Wtd Cons  |
| -------------- | ---- | ---- | ---------- | ---------- | --------- |
| `no_term`      | 39.6 | 65.2 | —          | —          | —         |
| original terms | 46.1 | 72.0 | 73.3       | 0.898      | 0.771     |
| + GPT-expanded | 53.9 | 77.5 | 82.5       | 0.948      | 0.843     |
| + Dictionary   | 54.2 | 77.7 | 79.1       | 0.937      | 0.828     |
| + GPT-cleaned  | 50.5 | 74.4 | **85.2**   | **0.973**  | **0.928** |

- Adding proper terms alone: **+6.5 BLEU / +6.8 chrF** over `no_term`.
- Expansion helps everything; **dictionary** edges out on BLEU/chrF, but **GPT-expansion** wins on terminology.
- **Cleaning** trades a little BLEU for the best term accuracy and consistency.

### 2. Fine-tuning: best models (macro-avg EN→DE/RU/ES)

| Model                    | BLEU     | chrF     | Term Acc % | Macro Cons | Wtd Cons  |
| ------------------------ | -------- | -------- | ---------- | ---------- | --------- |
| Qwen-7B base (+few-shot) | 38.4     | 66.2     | 67.8       | 0.892      | 0.760     |
| Qwen-7B LoRA (2 epochs)  | **51.4** | **75.0** | 69.9       | 0.896      | 0.772     |
| GPT-4o-mini              | 47.5     | 73.3     | **74.8**   | **0.900**  | **0.773** |

- LoRA lifts Qwen-7B by **+13 BLEU / +8.8 chrF** over the few-shot base.
- Tuned Qwen-7B **beats GPT-4o-mini on BLEU/chrF** (esp. EN→RU: 45.7 vs 37.2 BLEU).
- **GPT-4o-mini still leads on term accuracy & consistency.**

**Figure:** `figures/finetuning_gains.png` and `figures/model_comparison_heatmap.png` (optional support).

---

## Column 4 — Analysis & Conclusions

### Key Findings

1. **Terminology helps — but BLEU/chrF hide it.** General metrics react to term _quantity_; only term accuracy & consistency reveal whether terms are actually preserved.
2. **How you expand matters more than how much.** GPT in-context expansion is best for terminology fidelity; a dictionary from a related corpus mainly nudges BLEU/chrF. Cleaning to domain-specific terms gives the strongest terminology control overall.
3. **Split verdict on models.** Fine-tuned Qwen-7B (2 epochs) matches or beats GPT-4o-mini on general quality, yet **GPT-4o-mini remains the most reliable for correct, consistent proper-term translation.**

### Qualitative Examples (proper term in **bold**)

1. **EN→DE** — _"The status of each individual **space** can be seen from the color code…"_ → _"Der Farbcode oben links gibt den Status des betreffenden **Space** an."_ (`space → Space`)
2. **EN→RU** — _"…open the Notes **pane** on the left side of the screen."_ → _"…откройте **область** Примечания на левой стороне экрана."_ (`pane → область`)
3. **EN→ES** — _"Decide if you want to use **parallel processing** for this **job**:"_ → _"Decida si desea utilizar el **procesamiento paralelo** para este **job**:"_ (`parallel processing → procesamiento paralelo`, `job → job`)

### Limitations & Future Work

1. **Train–test lexical overlap.** Some dev_v1 test sentences closely resemble dev_v2 training data (overlap ratio > 0.5); the high-overlap subset is also generally easier, so gains are not purely memorization — but similarity remains a reporting caveat.
2. **Narrow scope.** Only 3 language pairs and a single (SAP/IT) domain; generalization to other domains/languages is untested.
3. **Small evaluation sets.** 497–1,500 sentences per language limit statistical power.
4. **Next steps:** extract terms at inference (no oracle terms), broaden domains/languages, and adopt terminology-aware metrics as primary selection criteria.

### References

1. Semenov, Huang, Zouhar, Berger, Zhu, Oncevay & Chen (2025). _Findings of the WMT25 Terminology Translation Task: Terminology is Useful Especially for Good MTs._ Proc. WMT25, ACL.
2. Exner et al. — SAP software-documentation MT corpus (`term_postedits`).

**QR code:** https://github.com/vnpnko/terminology-translation
