"""Build pandas DataFrames for finetuning experiment Excel sheets."""

from __future__ import annotations

import pandas as pd

from metrics_parser import (
    LANG_LABELS,
    LANG_ORDER,
    LangMetrics,
    ParsedRun,
    macro_average,
    parse_training_loss,
)

ROW_KEYS = (*LANG_ORDER, "MACRO_AVG")
ROW_LABELS = {**LANG_LABELS, "MACRO_AVG": "MACRO_AVG"}


def _row_metrics(by_lang: dict[str, LangMetrics], row_key: str) -> LangMetrics:
    if row_key == "MACRO_AVG":
        return macro_average(by_lang)
    return by_lang.get(row_key, LangMetrics())


def _delta(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return a - b


def _find_run(runs: list[ParsedRun], run_id: str) -> ParsedRun | None:
    for run in runs:
        if run.config.run_id == run_id:
            return run
    return None


def _lora_runs(runs: list[ParsedRun]) -> list[ParsedRun]:
    return [r for r in runs if not r.config.is_baseline]


def _best_lora_run(runs: list[ParsedRun], lang: str) -> ParsedRun | None:
    candidates: list[tuple[ParsedRun, float, float]] = []
    for run in _lora_runs(runs):
        metrics = _row_metrics(run.by_lang, lang)
        if metrics.term_avg_pct is not None:
            candidates.append((run, metrics.term_avg_pct, metrics.bleu or float("-inf")))

    if not candidates:
        return None

    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return candidates[0][0]


def build_main_results(
    runs: list[ParsedRun],
    gpt_run: ParsedRun | None,
) -> pd.DataFrame:
    base_run = _find_run(runs, "base")
    rows: list[dict[str, object]] = []

    for row_key in ROW_KEYS:
        row: dict[str, object] = {"lang_pair": ROW_LABELS[row_key]}
        gpt_metrics = _row_metrics(gpt_run.by_lang, row_key) if gpt_run else LangMetrics()
        base_metrics = _row_metrics(base_run.by_lang, row_key) if base_run else LangMetrics()

        for run in runs:
            rid = run.config.run_id
            m = _row_metrics(run.by_lang, row_key)
            row[f"{rid}_bleu"] = m.bleu
            row[f"{rid}_chrf"] = m.chrf
            row[f"{rid}_total_terms"] = m.total_terms
            row[f"{rid}_term_avg_pct"] = m.term_avg_pct

            row[f"d_{rid}_bleu_vs_gpt"] = _delta(m.bleu, gpt_metrics.bleu)
            row[f"d_{rid}_chrf_vs_gpt"] = _delta(m.chrf, gpt_metrics.chrf)
            row[f"d_{rid}_term_vs_gpt"] = _delta(m.term_avg_pct, gpt_metrics.term_avg_pct)

            if not run.config.is_baseline:
                row[f"d_{rid}_bleu_vs_base"] = _delta(m.bleu, base_metrics.bleu)
                row[f"d_{rid}_chrf_vs_base"] = _delta(m.chrf, base_metrics.chrf)
                row[f"d_{rid}_term_vs_base"] = _delta(m.term_avg_pct, base_metrics.term_avg_pct)

        rows.append(row)

    return pd.DataFrame(rows)


def build_base_vs_lora(runs: list[ParsedRun]) -> pd.DataFrame:
    base_run = _find_run(runs, "base")
    rows: list[dict[str, object]] = []

    for row_key in ROW_KEYS:
        base_m = _row_metrics(base_run.by_lang, row_key) if base_run else LangMetrics()

        if row_key == "MACRO_AVG":
            best_by_lang: dict[str, LangMetrics] = {}
            for lang in LANG_ORDER:
                br = _best_lora_run(runs, lang)
                if br:
                    best_by_lang[lang] = _row_metrics(br.by_lang, lang)
            best_m = macro_average(best_by_lang)
            best_config = ", ".join(
                sorted({_best_lora_run(runs, lang).config.run_id for lang in LANG_ORDER if _best_lora_run(runs, lang)})
            )
        else:
            best = _best_lora_run(runs, row_key)
            best_m = _row_metrics(best.by_lang, row_key) if best else LangMetrics()
            best_config = best.config.run_id if best else ""

        rows.append(
            {
                "lang_pair": ROW_LABELS[row_key],
                "base_bleu": base_m.bleu,
                "base_chrf": base_m.chrf,
                "base_term_avg_pct": base_m.term_avg_pct,
                "best_lora_bleu": best_m.bleu,
                "best_lora_chrf": best_m.chrf,
                "best_lora_term_avg_pct": best_m.term_avg_pct,
                "d_bleu": _delta(best_m.bleu, base_m.bleu),
                "d_chrf": _delta(best_m.chrf, base_m.chrf),
                "d_term_avg_pct": _delta(best_m.term_avg_pct, base_m.term_avg_pct),
                "best_config": best_config,
            }
        )

    return pd.DataFrame(rows)


def build_epoch_ablation(
    runs: list[ParsedRun],
    low_epoch_id: str = "lora_2ep_nofs",
    high_epoch_id: str = "lora_3ep_nofs",
) -> pd.DataFrame:
    low_run = _find_run(runs, low_epoch_id)
    high_run = _find_run(runs, high_epoch_id)
    rows: list[dict[str, object]] = []

    for row_key in ROW_KEYS:
        low_m = _row_metrics(low_run.by_lang, row_key) if low_run else LangMetrics()
        high_m = _row_metrics(high_run.by_lang, row_key) if high_run else LangMetrics()
        rows.append(
            {
                "lang_pair": ROW_LABELS[row_key],
                f"{low_epoch_id}_bleu": low_m.bleu,
                f"{high_epoch_id}_bleu": high_m.bleu,
                "d_bleu": _delta(high_m.bleu, low_m.bleu),
                f"{low_epoch_id}_chrf": low_m.chrf,
                f"{high_epoch_id}_chrf": high_m.chrf,
                "d_chrf": _delta(high_m.chrf, low_m.chrf),
                f"{low_epoch_id}_term_avg_pct": low_m.term_avg_pct,
                f"{high_epoch_id}_term_avg_pct": high_m.term_avg_pct,
                "d_term_avg_pct": _delta(high_m.term_avg_pct, low_m.term_avg_pct),
            }
        )

    return pd.DataFrame(rows)


def build_training_loss(runs: list[ParsedRun]) -> tuple[pd.DataFrame, pd.DataFrame]:
    series: dict[str, pd.DataFrame] = {}
    meta_rows: list[dict[str, object]] = []

    for run in runs:
        path = run.config.training_loss_path
        if path is None:
            df = pd.DataFrame(columns=["step", "loss"])
        else:
            df = parse_training_loss(path)

        if not df.empty:
            series[run.config.run_id] = df.rename(columns={"loss": run.config.run_id})

        final_loss = df["loss"].iloc[-1] if not df.empty else None
        final_step = int(df["step"].iloc[-1]) if not df.empty else None
        meta_rows.append(
            {
                "run_id": run.config.run_id,
                "num_epochs": run.config.num_epochs,
                "training_loss_exists": run.config.training_loss_exists,
                "training_loss_populated": run.config.training_loss_populated,
                "final_step": final_step,
                "final_loss": final_loss,
            }
        )

    if not series:
        wide = pd.DataFrame(columns=["step"])
    else:
        wide = None
        for run_id, df in series.items():
            part = df[["step", run_id]]
            wide = part if wide is None else wide.merge(part, on="step", how="outer")
        wide = wide.sort_values("step").reset_index(drop=True)

    return wide, pd.DataFrame(meta_rows)


def build_experiment_config(runs: list[ParsedRun]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run in runs:
        cfg = run.config
        rows.append(
            {
                "run_id": cfg.run_id,
                "label": cfg.label,
                "folder": cfg.folder,
                "model": cfg.model,
                "use_few_shot": cfg.use_few_shot,
                "num_epochs": cfg.num_epochs,
                "mode": cfg.mode,
                "metrics_file_exists": cfg.metrics_file_exists,
                "training_loss_exists": cfg.training_loss_exists,
                "training_loss_populated": cfg.training_loss_populated,
                "metrics_path": str(cfg.metrics_path) if cfg.metrics_path else "",
                "notes": cfg.notes,
            }
        )
    return pd.DataFrame(rows)


def build_gpt_baseline(gpt_run: ParsedRun) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row_key in ROW_KEYS:
        m = _row_metrics(gpt_run.by_lang, row_key)
        rows.append(
            {
                "lang_pair": ROW_LABELS[row_key],
                "bleu": m.bleu,
                "chrf": m.chrf,
                "total_terms": m.total_terms,
                "term_avg_pct": m.term_avg_pct,
                "sample_count": m.sample_count,
            }
        )
    return pd.DataFrame(rows)


def build_gpt_vs_best_qwen(
    gpt_run: ParsedRun,
    qwen_runs_by_size: dict[str, list[ParsedRun]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for row_key in LANG_ORDER:
        gpt_m = _row_metrics(gpt_run.by_lang, row_key)
        for size, runs in qwen_runs_by_size.items():
            best = _best_lora_run(runs, row_key)
            best_m = _row_metrics(best.by_lang, row_key) if best else LangMetrics()
            rows.append(
                {
                    "lang_pair": ROW_LABELS[row_key],
                    "qwen_size": size,
                    "gpt_bleu": gpt_m.bleu,
                    "best_qwen_bleu": best_m.bleu,
                    "d_bleu": _delta(best_m.bleu, gpt_m.bleu),
                    "gpt_chrf": gpt_m.chrf,
                    "best_qwen_chrf": best_m.chrf,
                    "d_chrf": _delta(best_m.chrf, gpt_m.chrf),
                    "gpt_term_avg_pct": gpt_m.term_avg_pct,
                    "best_qwen_term_avg_pct": best_m.term_avg_pct,
                    "d_term_avg_pct": _delta(best_m.term_avg_pct, gpt_m.term_avg_pct),
                    "best_qwen_config": best.config.run_id if best else "",
                }
            )

    return pd.DataFrame(rows)
