#!/usr/bin/env python3
"""
Fill ``random_terms`` in JSONL files that already have ``en``, a target post-edit, and ``proper_terms``.

For each line:
  1. An LLM picks 1–2 **random** (non-domain) word pairs in the sentence pair.
  2. Keys use the exact English surface form (substring of ``en``).
  3. Values are aligned to the target post-edit (substring of ``de`` / ``es`` / ``ru``).
  4. Terms must not duplicate keys already in ``proper_terms``.

Uses OpenRouter (same setup as ``fill_proper_terms_parallel.py``):
  OPENROUTER_API_KEY in ``scripts/.env``.

Usage::

    python fill_random_terms.py -t de -i input.jsonl -o output.jsonl
    python fill_random_terms.py -t ru -i enru_with_proper_terms.jsonl -o filled.jsonl
    python fill_random_terms.py -t es -i enes_dev.jsonl -o enes_dev.filled.jsonl --workers 1
    python fill_random_terms.py -t de -i input.jsonl -o output.jsonl --merge-only
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
DEFAULT_WORKERS = 10
DEFAULT_CHUNKS = 10
ENV_FILE = SCRIPTS_DIR / ".env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_MODEL = "openai/gpt-4o-mini"
MAX_RANDOM_TERMS = 2
MIN_RANDOM_TERMS = 1

EXAMPLE_PAIRS_DE = """
{"en": "Lock and Edit a Release\\n", "de": "Sperren und Bearbeiten eines Release\\n", "proper_terms": {"release": "Release"}, "random_terms": {"Lock": "Sperren"}}
{"en": "Open the consumption model containing the measures and attributes you want to include in your perspective, and click the  Perspectives tab.\\n", "de": "Öffnen Sie das Verbrauchsmodell mit den Kennzahlen und Attribute, die Sie in Ihre Perspektive aufnehmen möchten, un wechseln Sie zur Registerkarte Perspektiven.\\n", "proper_terms": {"consumption model": "Verbrauchsmodell"}, "random_terms": {"Open": "Öffnen", "Perspectives": "Perspektiven"}}
{"en": "This service describes the deployed (run-time) state of SAP HANA database artifacts, for example: tables, views, or procedures, which have been created or adjusted by the SAP Integrated Development Environment (WebIDE) editors as a family of consistent design-time artifacts for all key SAP HANA platform database features.\\n", "de": "Dieser Service beschreibt den implementierten Zustand (Laufzeitzustand) von SAP-HANA-Datenbankartefakten, z. B. Tabellen, Views oder Prozeduren, die von den SAP-Integrated-Development-Environment-Editoren (WebIDE-Editoren) als eine Familie konsistenter Entwurfszeit-Artefakte für alle wichtigen SAP-HANA-Plattform-Datenbankfunktionen erstellt oder angepasst wurden.\\n", "proper_terms": {"design": "Entwurf", "state": "Zustand"}, "random_terms": {"artifacts": "Artefakten", "key": "wichtigen"}}
""".strip()

EXAMPLE_PAIRS_ES = """
{"en": "In such cases you may use the Move Items or Merge feature.", "es": "En estos casos, puede utilizar la función Mover elementos o Fusionar .", "proper_terms": {"item": "elemento"}, "random_terms": {"cases": "casos", "use": "utilizar"}}
{"en": "Choose any text block within the Template Structure tab, to add a new condition for it or view the existing condition.", "es": "Seleccione cualquier bloque de texto dentro de la pestaña Estructura de plantilla para añadir una condición nueva o ver la condición existente.", "proper_terms": {"tab": "pestaña"}, "random_terms": {"Choose": "Seleccione", "new": "nueva"}}
""".strip()

EXAMPLE_PAIRS_RU = """
{"en": "Indicates if a configuration item or configuration step is specific to a localized solution version.", "ru": "Указывает, являются ли позиция или шаг конфигурации специфичными для локализованной версии решения.", "proper_terms": {"item": "позиция"}, "random_terms": {"localized": "локализованной", "specific": "специфичными"}}
{"en": "You run allocation cycles in the Run Allocations app.", "ru": "Для выполнения циклов перерасчета используется приложение Выполнить перерасчеты.", "proper_terms": {"allocation": "перерасчета"}, "random_terms": {"cycles": "циклов", "app": "приложение"}}
""".strip()


@dataclass(frozen=True)
class TargetLang:
    code: str
    name: str
    field: str
    examples: str


TARGET_LANGS: dict[str, TargetLang] = {
    "de": TargetLang("de", "German", "de", EXAMPLE_PAIRS_DE),
    "es": TargetLang("es", "Spanish", "es", EXAMPLE_PAIRS_ES),
    "ru": TargetLang("ru", "Russian", "ru", EXAMPLE_PAIRS_RU),
}


def load_dotenv(path: Path = ENV_FILE) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].strip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip().lstrip("\ufeff")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def locate_substring(needle: str, haystack: str) -> str | None:
    needle = needle.strip()
    if not needle:
        return None
    pattern = re.escape(needle)
    match = re.search(pattern, haystack, flags=re.IGNORECASE)
    if not match:
        return None
    return haystack[match.start() : match.end()]


def cap_terms(terms: dict[str, str], max_count: int = MAX_RANDOM_TERMS) -> dict[str, str]:
    if len(terms) <= max_count:
        return terms
    ranked = sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)
    return dict(ranked[:max_count])


def align_terms(
    en: str,
    target: str,
    raw_terms: dict[str, str],
    *,
    max_count: int = MAX_RANDOM_TERMS,
) -> dict[str, str]:
    aligned: dict[str, str] = {}
    for en_term, target_term in raw_terms.items():
        if not isinstance(en_term, str) or not isinstance(target_term, str):
            continue
        en_span = locate_substring(en_term, en)
        target_span = locate_substring(target_term, target)
        if en_span is None or target_span is None:
            continue
        if en_span in aligned:
            continue
        aligned[en_span] = target_span
    return cap_terms(aligned, max_count)


def proper_en_spans(proper_terms: dict[str, str]) -> set[str]:
    return {k.casefold() for k in proper_terms if isinstance(k, str)}


def drop_proper_overlaps(terms: dict[str, str], proper_terms: dict[str, str]) -> dict[str, str]:
    """Remove random terms whose English key matches a proper_terms key."""
    blocked = proper_en_spans(proper_terms)
    out: dict[str, str] = {}
    for en_span, de_span in terms.items():
        if en_span.casefold() in blocked:
            continue
        out[en_span] = de_span
    return out


def safe_print(text: str, *, file=None) -> None:
    """Print without crashing on Windows consoles that lack Unicode (e.g. cp1252)."""
    out = file or sys.stdout
    enc = getattr(out, "encoding", None) or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, file=out, flush=True)


def build_prompt(en: str, target: str, proper_terms: dict[str, str], lang: TargetLang) -> str:
    proper_json = json.dumps(proper_terms, ensure_ascii=False)
    return f"""You annotate SAP / enterprise IT documentation sentence pairs with **random terms**.

``proper_terms`` are already filled with domain / IT vocabulary. Your task is to add **random_terms**:
- Pick between {MIN_RANDOM_TERMS} and {MAX_RANDOM_TERMS} ordinary words (verbs, adjectives, generic nouns, UI verbs like Lock/Save/Open).
- NOT domain-specific SAP/IT terms (those belong in proper_terms).
- Do NOT reuse any English key already listed in proper_terms.
- Each English key MUST be copied exactly as it appears in the English sentence (surface form, including capitalization).
- Each {lang.name} value MUST be copied exactly as it appears in the {lang.name} sentence.
- Prefer words that clearly translate differently between EN and {lang.name} (not identical loanwords unless unavoidable).
- If no suitable random term exists, return an empty object: {{"random_terms": {{}}}}.

Return JSON only:
{{"random_terms": {{"<english surface>": "<{lang.name.lower()} surface>", ...}}}}

Examples:
{lang.examples}

proper_terms (do not repeat these English keys):
{proper_json}

English:
{en}

{lang.name}:
{target}
"""


def parse_model_json(content: str) -> dict[str, Any]:
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


def openrouter_chat(
    *,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    timeout: float,
) -> str:
    body = json.dumps(
        {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "response_format": {"type": "json_object"},
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {payload}") from exc


def extract_random_terms(
    en: str,
    target: str,
    proper_terms: dict[str, str],
    lang: TargetLang,
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
) -> dict[str, str]:
    prompt = build_prompt(en, target, proper_terms, lang)
    messages = [
        {"role": "system", "content": "You respond with valid JSON only."},
        {"role": "user", "content": prompt},
    ]

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            raw = openrouter_chat(
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                timeout=timeout,
            )
            parsed = parse_model_json(raw)
            break
        except (
            urllib.error.URLError,
            urllib.error.HTTPError,
            json.JSONDecodeError,
            RuntimeError,
        ) as exc:
            last_error = exc
            if attempt == max_retries:
                raise
            time.sleep(min(2**attempt, 30))
    else:
        raise last_error  # pragma: no cover

    model_terms = parsed.get("random_terms") or {}
    if not isinstance(model_terms, dict):
        raise ValueError(f"Invalid random_terms in model response: {model_terms!r}")

    aligned = drop_proper_overlaps(align_terms(en, target, model_terms), proper_terms)
    if len(aligned) >= MIN_RANDOM_TERMS:
        return aligned

    retry_messages = messages + [
        {
            "role": "user",
            "content": (
                "Your previous answer had no valid random-term pairs (or overlapped proper_terms). "
                f"Return {MIN_RANDOM_TERMS} or {MAX_RANDOM_TERMS} pairs of generic words; "
                f"both spans must appear verbatim in the EN and {lang.name} sentences; "
                "do not repeat proper_terms keys."
            ),
        }
    ]
    raw = openrouter_chat(
        api_key=api_key,
        model=model,
        messages=retry_messages,
        temperature=temperature,
        timeout=timeout,
    )
    parsed = parse_model_json(raw)
    model_terms = parsed.get("random_terms") or {}
    if not isinstance(model_terms, dict):
        raise ValueError(f"Invalid random_terms on retry: {model_terms!r}")
    aligned = drop_proper_overlaps(align_terms(en, target, model_terms), proper_terms)
    if len(aligned) < MIN_RANDOM_TERMS:
        return {}
    return aligned


def record_needs_fill(record: dict) -> bool:
    random = record.get("random_terms") or {}
    if not isinstance(random, dict) or not random:
        return True
    return any(not isinstance(v, str) or not v.strip() for v in random.values())


def process_file(
    input_path: Path,
    output_path: Path,
    lang: TargetLang,
    *,
    api_key: str,
    model: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    delay: float,
    limit: int | None,
    force: bool,
    dry_run: bool,
    save_every: int,
    require_proper: bool,
) -> tuple[int, int]:
    records: list[dict] = []
    with input_path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            records.append(json.loads(raw))

    if not records:
        safe_print(f"Skip empty file: {input_path}")
        return 0, 0

    updated = 0
    skipped = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        if dry_run:
            return
        lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
        tmp = output_path.with_suffix(output_path.suffix + ".tmp")
        tmp.write_text("".join(lines), encoding="utf-8")
        tmp.replace(output_path)

    for idx, record in enumerate(records):
        if limit is not None and updated >= limit:
            break
        if not force and not record_needs_fill(record):
            skipped += 1
            continue

        en = record.get("en", "")
        target = record.get(lang.field, "")
        if (
            not isinstance(en, str)
            or not isinstance(target, str)
            or not en.strip()
            or not target.strip()
        ):
            safe_print(
                f"  [{input_path.name}:{idx + 1}] skip: missing en/{lang.field}",
                file=sys.stderr,
            )
            skipped += 1
            continue

        proper_terms = record.get("proper_terms") or {}
        if not isinstance(proper_terms, dict):
            proper_terms = {}
        if require_proper and not proper_terms:
            safe_print(
                f"  [{input_path.name}:{idx + 1}] skip: empty proper_terms",
                file=sys.stderr,
            )
            skipped += 1
            continue

        if dry_run:
            safe_print(f"  [{input_path.name}:{idx + 1}] would fill random_terms")
            updated += 1
            continue

        random_terms = extract_random_terms(
            en,
            target,
            proper_terms,
            lang,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        out = dict(record)
        out["random_terms"] = random_terms
        records[idx] = out
        updated += 1
        status = "(no terms)" if not random_terms else repr(random_terms)
        safe_print(f"  [{input_path.name}:{idx + 1}/{len(records)}] {status}")

        if save_every > 0 and updated % save_every == 0:
            flush()

        if delay > 0:
            time.sleep(delay)

    if updated and not dry_run:
        flush()

    return updated, skipped


def read_nonempty_lines(path: Path) -> list[str]:
    lines: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if raw.strip():
                lines.append(raw)
    return lines


def split_lines(lines: list[str], num_chunks: int) -> list[list[str]]:
    if not lines:
        return []
    if num_chunks < 1:
        raise ValueError("num_chunks must be >= 1")
    n = len(lines)
    base, remainder = divmod(n, num_chunks)
    chunks: list[list[str]] = []
    start = 0
    for i in range(num_chunks):
        size = base + (1 if i < remainder else 0)
        if size == 0:
            continue
        chunks.append(lines[start : start + size])
        start += size
    return chunks


def write_chunks(lines: list[str], num_chunks: int, work_dir: Path) -> list[Path]:
    work_dir.mkdir(parents=True, exist_ok=True)
    chunks = split_lines(lines, num_chunks)
    paths: list[Path] = []
    for i, chunk_lines in enumerate(chunks):
        path = work_dir / f"chunk_{i:02d}.jsonl"
        text = "\n".join(chunk_lines) + ("\n" if chunk_lines else "")
        path.write_text(text, encoding="utf-8")
        paths.append(path)
        print(f"  wrote {path.name}: {len(chunk_lines)} lines")
    return paths


def merge_chunks(chunk_outputs: list[Path], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with output_path.open("w", encoding="utf-8", newline="\n") as out:
        for path in chunk_outputs:
            if not path.is_file():
                raise FileNotFoundError(f"Missing chunk output: {path}")
            with path.open(encoding="utf-8") as f:
                for line in f:
                    raw = line.rstrip("\n")
                    if not raw.strip():
                        continue
                    out.write(raw + "\n")
                    total += 1
    return total


def build_chunk_cmd(
    chunk_in: Path,
    chunk_out: Path,
    *,
    target_lang: str,
    model: str,
    api_key: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    delay: float,
    save_every: int,
    force: bool,
    require_proper: bool,
    dry_run: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "-t",
        target_lang,
        "-i",
        str(chunk_in),
        "-o",
        str(chunk_out),
        "--workers",
        "1",
        "--model",
        model,
        "--temperature",
        str(temperature),
        "--timeout",
        str(timeout),
        "--max-retries",
        str(max_retries),
        "--delay",
        str(delay),
        "--save-every",
        str(save_every),
    ]
    if api_key:
        cmd.extend(["--api-key", api_key])
    if force:
        cmd.append("--force")
    if require_proper:
        cmd.append("--require-proper")
    if dry_run:
        cmd.append("--dry-run")
    return cmd


def run_fill_chunk(chunk_idx: int, cmd: list[str], chunk_out: Path) -> tuple[int, Path]:
    print(f"[chunk {chunk_idx:02d}] start -> {chunk_out.name}", flush=True)
    proc = subprocess.run(
        cmd,
        cwd=SCRIPT_DIR,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if proc.stdout:
        safe_print(proc.stdout)
    if proc.returncode != 0:
        err = proc.stderr or proc.stdout or "(no output)"
        raise RuntimeError(f"chunk {chunk_idx:02d} failed (exit {proc.returncode}):\n{err}")
    if proc.stderr:
        safe_print(proc.stderr, file=sys.stderr)
    print(f"[chunk {chunk_idx:02d}] done", flush=True)
    return chunk_idx, chunk_out


def run_parallel(
    input_path: Path,
    output_path: Path,
    lang: TargetLang,
    *,
    work_dir: Path,
    chunks: int,
    workers: int,
    clean: bool,
    merge_only: bool,
    model: str,
    api_key: str,
    temperature: float,
    timeout: float,
    max_retries: int,
    delay: float,
    save_every: int,
    force: bool,
    require_proper: bool,
    dry_run: bool,
) -> None:
    lines = read_nonempty_lines(input_path)
    print(f"Input: {input_path} ({len(lines)} lines)")
    print(f"Target language: {lang.code} ({lang.name})")
    print(f"Chunks: {chunks}, workers: {workers}, work-dir: {work_dir}")

    chunk_out_dir = work_dir / "out"
    existing_outputs = sorted(chunk_out_dir.glob("chunk_*.filled.jsonl"))

    if merge_only:
        if not existing_outputs:
            raise SystemExit(f"No chunk outputs in {chunk_out_dir}")
        print(f"Merging {len(existing_outputs)} chunk(s) -> {output_path} ...")
        total = merge_chunks(existing_outputs, output_path)
        if total != len(lines):
            raise SystemExit(
                f"Line count mismatch after merge: expected {len(lines)}, got {total}"
            )
        print(f"Done: merged {total} lines into {output_path}")
        return

    if clean and work_dir.exists():
        shutil.rmtree(work_dir)
        print(f"Removed {work_dir}")

    print("Splitting...")
    chunk_inputs = write_chunks(lines, chunks, work_dir / "in")
    chunk_outputs = [
        chunk_out_dir / f"chunk_{i:02d}.filled.jsonl" for i in range(len(chunk_inputs))
    ]
    chunk_out_dir.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print("Dry run: would process each chunk with --workers 1, then merge.")
        return

    print(f"Processing {len(chunk_inputs)} chunks with {workers} workers...")
    results: dict[int, Path] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                run_fill_chunk,
                i,
                build_chunk_cmd(
                    chunk_in,
                    chunk_out,
                    target_lang=lang.code,
                    model=model,
                    api_key=api_key,
                    temperature=temperature,
                    timeout=timeout,
                    max_retries=max_retries,
                    delay=delay,
                    save_every=save_every,
                    force=force,
                    require_proper=require_proper,
                    dry_run=False,
                ),
                chunk_out,
            )
            for i, (chunk_in, chunk_out) in enumerate(zip(chunk_inputs, chunk_outputs))
        ]
        for fut in as_completed(futures):
            idx, path = fut.result()
            results[idx] = path

    ordered_outputs = [results[i] for i in range(len(chunk_inputs))]
    print(f"Merging -> {output_path} ...")
    total = merge_chunks(ordered_outputs, output_path)
    if total != len(lines):
        raise SystemExit(
            f"Line count mismatch after merge: expected {len(lines)}, got {total}"
        )
    print(f"Done: merged {total} lines into {output_path}")
    print(f"Chunk files kept in {work_dir}")


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Fill random_terms in JSONL via LLM (parallel by default)."
    )
    parser.add_argument(
        "-t",
        "--target-lang",
        required=True,
        choices=sorted(TARGET_LANGS),
        help="Target language code: de (German), es (Spanish), ru (Russian)",
    )
    parser.add_argument("-i", "--input", type=Path, required=True)
    parser.add_argument("-o", "--output", type=Path, required=True)
    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=f"Parallel chunk workers (default: {DEFAULT_WORKERS}; use 1 for sequential)",
    )
    parser.add_argument(
        "--chunks",
        type=int,
        default=DEFAULT_CHUNKS,
        help=f"Number of input chunks (default: {DEFAULT_CHUNKS})",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=None,
        help="Chunk work directory (default: <output_stem>.chunks/)",
    )
    parser.add_argument("--clean", action="store_true", help="Delete work-dir before run")
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Only merge existing chunk outputs",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--delay", type=float, default=0.5)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-proper",
        action="store_true",
        help="Skip lines with empty proper_terms",
    )
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()

    if not input_path.is_file():
        raise SystemExit(f"Input file not found: {input_path}")

    lang = TARGET_LANGS[args.target_lang]

    if not args.dry_run and not args.merge_only and not args.api_key:
        raise SystemExit(
            f"Missing API key. Set OPENROUTER_API_KEY in {ENV_FILE} or pass --api-key."
        )

    if args.workers > 1 or args.merge_only:
        work_dir = (
            args.work_dir.resolve()
            if args.work_dir
            else output_path.parent / f"{output_path.stem}.chunks"
        )
        if args.chunks < 1:
            raise SystemExit("--chunks must be >= 1")
        if args.workers < 1:
            raise SystemExit("--workers must be >= 1")
        run_parallel(
            input_path,
            output_path,
            lang,
            work_dir=work_dir,
            chunks=args.chunks,
            workers=args.workers,
            clean=args.clean,
            merge_only=args.merge_only,
            model=args.model,
            api_key=args.api_key,
            temperature=args.temperature,
            timeout=args.timeout,
            max_retries=args.max_retries,
            delay=args.delay,
            save_every=args.save_every,
            force=args.force,
            require_proper=args.require_proper,
            dry_run=args.dry_run,
        )
        return

    safe_print(f"Processing {input_path} -> {output_path} ({lang.name}) ...")
    updated, skipped = process_file(
        input_path,
        output_path,
        lang,
        api_key=args.api_key,
        model=args.model,
        temperature=args.temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        delay=args.delay,
        limit=args.limit,
        force=args.force,
        dry_run=args.dry_run,
        save_every=args.save_every,
        require_proper=args.require_proper,
    )
    safe_print(f"Done: {updated} filled, {skipped} skipped")


if __name__ == "__main__":
    main()
