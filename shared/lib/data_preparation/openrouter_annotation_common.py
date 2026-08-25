"""Shared OpenRouter chunking, parallel-request, and dotenv scaffolding for the annotate_*_openrouter.py scripts."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
ENV_FILE = SCRIPTS_DIR / ".env"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


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
        key = key.strip().lstrip("﻿")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value


def safe_print(text: str, *, file: Any = None) -> None:
    """Print without crashing on Windows consoles that lack Unicode (e.g. cp1252)."""
    out = file or sys.stdout
    enc = getattr(out, "encoding", None) or "utf-8"
    safe = text.encode(enc, errors="replace").decode(enc, errors="replace")
    print(safe, file=out, flush=True)


def locate_substring(needle: str, haystack: str) -> str | None:
    needle = needle.strip()
    if not needle:
        return None
    pattern = re.escape(needle)
    match = re.search(pattern, haystack, flags=re.IGNORECASE)
    if not match:
        return None
    return haystack[match.start() : match.end()]


def cap_terms(terms: dict[str, str], max_count: int) -> dict[str, str]:
    if len(terms) <= max_count:
        return terms
    ranked = sorted(terms.items(), key=lambda kv: len(kv[0]), reverse=True)
    return dict(ranked[:max_count])


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
        safe_print(f"  wrote {path.name}: {len(chunk_lines)} lines")
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


def run_fill_chunk(chunk_idx: int, cmd: list[str], chunk_out: Path) -> tuple[int, Path]:
    safe_print(f"[chunk {chunk_idx:02d}] start -> {chunk_out.name}")
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
    safe_print(f"[chunk {chunk_idx:02d}] done")
    return chunk_idx, chunk_out
