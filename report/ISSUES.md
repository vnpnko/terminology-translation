# Known gaps and open questions

Things about the current repo state worth knowing before writing the paper. Companion to [`README.md`](README.md).

## `gpt_proposed_term_pipeline` needs review before it's paper-ready

[`experiments/term_expansion/gpt_proposed_term_pipeline/`](../experiments/term_expansion/gpt_proposed_term_pipeline/README.md) (oracle `proper_term` vs. GPT self-proposed `gpt_proposed_term`, zero-shot, `dev_v1/original`) was recently restored/ported after having been removed from the repo. Open before it's cited anywhere:
- Its cached prediction data predates the current repo structure and hasn't been re-validated since restoration — see the experiment README's "Restored/ported work" note.
- It has no figure script (Excel-only), which is part of why it was removed the first time — visualization is still an open design question.
- It isn't yet reflected in this doc's Experiments section outline below (no §3.x entry) — whether/how it becomes a paper RQ is a decision for after review, not assumed here.
