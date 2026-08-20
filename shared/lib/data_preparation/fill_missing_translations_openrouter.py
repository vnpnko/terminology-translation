"""Fill target-language sentences and term translations in JSONL dev files.

Writes back in place (files resolved against ``--input-dir``, default
``data/sap_dev/``, which doesn't exist in this repo; pass ``--input-dir``
to point at real data) via an atomic temp-file replace; pass ``--dry-run`` to
preview without writing. Uses GPT-4o-mini via OpenRouter
(https://openrouter.ai); set ``OPENROUTER_API_KEY`` in ``.env`` at the repo
root (see ``.env.example``).

Usage::

    python fill_missing_translations_openrouter.py ende_dev_1.jsonl
    python fill_missing_translations_openrouter.py ende_dev_1.jsonl --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from fill_missing_translations_openrouter_prompt import (
    TERM_FIELDS,
    translate_record,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ENV_FILE = REPO_ROOT / ".env"
DEFAULT_INPUT_DIR = REPO_ROOT / "shared" / "data" / "sap_dev"
DEFAULT_MODEL = "openai/gpt-4o-mini"

LANG_PAIR_PREFIXES: dict[str, tuple[str, str]] = {
    "ende": ("de", "German"),
    "enes": ("es", "Spanish"),
    "enru": ("ru", "Russian"),
}


def load_dotenv(path: Path = ENV_FILE) -> None:
    """Load KEY=VALUE lines from the repo-root .env (does not override existing env vars)."""
    if not path.is_file():
        return
    # utf-8-sig strips a Windows UTF-8 BOM so the variable name is not corrupted
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


def detect_language(path: Path, record: dict[str, Any]) -> tuple[str, str]:
    stem = path.stem.lower()
    for prefix, pair in LANG_PAIR_PREFIXES.items():
        if stem.startswith(prefix):
            return pair
    for code, name in (("de", "German"), ("es", "Spanish"), ("ru", "Russian")):
        if code in record and code != "en":
            return code, name
    raise ValueError(f"Cannot detect target language for {path}")


def record_needs_translation(record: dict[str, Any], tgt_code: str) -> bool:
    sentence = record.get(tgt_code, "")
    if not isinstance(sentence, str) or not sentence.strip():
        return True
    for field in TERM_FIELDS:
        terms = record.get(field) or {}
        for value in terms.values():
            if not isinstance(value, str) or not value.strip():
                return True
    return False


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    lines = [json.dumps(r, ensure_ascii=False) + "\n" for r in records]
    tmp.write_text("".join(lines), encoding="utf-8")
    tmp.replace(path)


def process_file(
    path: Path,
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
) -> tuple[int, int]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            raw = line.rstrip("\n")
            if not raw.strip():
                continue
            records.append(json.loads(raw))

    if not records:
        print(f"Skip empty file: {path}")
        return 0, 0

    tgt_code, target_lang = detect_language(path, records[0])
    updated = 0
    skipped = 0

    for idx, record in enumerate(records):
        if limit is not None and updated >= limit:
            break
        if not force and not record_needs_translation(record, tgt_code):
            skipped += 1
            continue

        if dry_run:
            print(f"  [{path.name}:{idx + 1}] would translate -> {target_lang}")
            updated += 1
            continue

        records[idx] = translate_record(
            record,
            tgt_code=tgt_code,
            target_lang=target_lang,
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout,
            max_retries=max_retries,
        )
        updated += 1
        print(f"  [{path.name}:{idx + 1}/{len(records)}] translated")

        if save_every > 0 and updated % save_every == 0:
            write_jsonl(path, records)

        if delay > 0:
            time.sleep(delay)

    if not dry_run and updated > 0:
        write_jsonl(path, records)

    return updated, skipped


def resolve_input_paths(names: list[str], input_dir: Path) -> list[Path]:
    paths: list[Path] = []
    for name in names:
        candidate = Path(name)
        if candidate.is_file():
            paths.append(candidate.resolve())
            continue
        in_data_dir = input_dir / name
        if in_data_dir.is_file():
            paths.append(in_data_dir.resolve())
            continue
        raise FileNotFoundError(
            f"File not found: {name!r} (also checked {in_data_dir})"
        )
    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill translations in new data/ JSONL files via OpenRouter.",
        epilog="Example: python fill_translations.py ende_dev_1.jsonl",
    )
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="JSONL filename(s), looked up in new data/ unless a path is given",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Folder to resolve FILE names (default: {DEFAULT_INPUT_DIR.name}/)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"OpenRouter model id (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("OPENROUTER_API_KEY", ""),
        help="OpenRouter API key (default: repo-root .env or OPENROUTER_API_KEY)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Sampling temperature (default: 0.2)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="HTTP timeout in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per record on API/parse errors (default: 3)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Seconds to wait between API calls (default: 0.5)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Write checkpoint every N translations (default: 10, 0=only at end)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max records to translate per file (for testing)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-translate even when target fields are already filled",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List work without calling the API or writing files",
    )
    return parser.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")
    paths = resolve_input_paths(args.files, input_dir)

    if not args.dry_run and not args.api_key:
        raise SystemExit(
            f"Missing API key. Set OPENROUTER_API_KEY in {ENV_FILE} "
            "(see .env.example) or pass --api-key."
        )

    total_updated = 0
    total_skipped = 0
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
        print(f"Processing {path} ...")
        updated, skipped = process_file(
            path,
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
        )
        total_updated += updated
        total_skipped += skipped
        print(f"  done: {updated} translated, {skipped} skipped")

    print(
        f"Finished: {total_updated} translated, {total_skipped} skipped "
        f"across {len(paths)} file(s)."
    )


if __name__ == "__main__":
    main()
