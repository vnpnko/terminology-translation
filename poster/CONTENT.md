# Poster Content — Single Source of Truth

---

## Header

**Title:** Terminology-Constrained Machine Translation for SAP Documentation

**Subtitle:** EN→{DE, RU, ES} with prompt conditioning, GPT pipeline, and LoRA fine-tuning

**Authors:** Ivan Pylypenko

**Affiliation:** Technical University of Munich — Practical Course

---

## Column 1 — Motivation & Task

### Problem

- Enterprise MT must preserve SAP/IT domain terms (UI labels, product names, technical concepts).
- Generic MT often paraphrases or mistranslates terminology, producing inconsistent product documentation.

### Research Questions

- Does explicit terminology in the prompt improve BLEU, chrF, term accuracy and and consistency?
- Does LoRA fine-tuning on SAP data close the gap for open models (Qwen)?

### Data & Task

- **dev_v1:** 500 sentences per language (course evaluation set).
- **dev_v2:** ~1,500 sentences per language from the [SAP term_postedits](https://github.com/SAP/software-documentation-data-set-for-machine-translation/tree/master/term_postedits/) corpus.
- Each JSONL record: English source, target reference, `proper_terms` (1–2 IT term pairs), `random_terms` (control).
- Language pairs: EN→DE, EN→RU, EN→ES.

---

## Column 2 — Methods

### Prompt-Conditioning Modes

| Mode | Description |
|------|-------------|
| `no_term` | Translate without terminology |
| `proper_term` | Inject oracle domain term pairs |
| `random_term` | Inject non-domain control pairs |

### Models

- **GPT-4o-mini** (OpenRouter) — primary baseline
- **Qwen2.5-3B / 7B-Instruct** — open-model baseline
- 3 hardcoded few-shot examples per language

### GPT 3-Step Pipeline

1. Extract EN domain terms (no reference)
2. Propose target translations (no reference)
3. Translate with proposed terms → `gpt_proposed_term`

### LoRA Fine-Tuning

- Train on dev_v2 (1500/lang); test on held-out dev_v1 (497/lang).
- Qwen 3B/7B: base, LoRA 1–3 epochs, with/without few-shot.
- Best config highlighted: **7B LoRA 2 epochs, no few-shot**.

---

## Column 3 — Results

1. ...
2. ...
3. ...

---

## Column 4 — Analysis & Conclusions

### Key Findings

1. `proper_term` consistently beats `no_term` and `random_term` — domain terms matter.
2. GPT-4o-mini leads on BLEU and chrF.
3. ...

### Qualitative Examples

1. ...
2. ...
3. ...

### Limitations & Future Work

1. ...
2. ...
3. ...

### References

1. Exner et al. — SAP software documentation MT corpus (term_postedits).
2. ...
3. ...

**QR code:** https://github.com/vnpnko/terminology-translation
