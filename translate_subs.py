#!/usr/bin/env python3
"""Translate .ASS or .SRT subtitles using a local Ollama model.

Workflow:
1) Read subtitles and build a brief summary for context.
2) Translate lines (SRT) or dialogue text segments (ASS), preserving tags.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager
import fnmatch
import json
import math
import os
import socket
import subprocess
import re
import sys
import time
from pathlib import Path
from typing import Any, List, Tuple, TypeVar
from urllib import request, error

try:
    from rich.console import Console
    from rich.progress import (
        BarColumn,
        Progress,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


ASS_TAG_RE = re.compile(r"(\{[^}]*\}|\\N|\\n|\\h)")
ASS_STRIP_RE = re.compile(r"\{[^}]*\}")
PLACEHOLDER_TOKEN_RE = re.compile(r"__TAG_\d+__")
LEGACY_PLACEHOLDER_RE = re.compile(r"__ASS_TAG_\d+__")
BROKEN_PLACEHOLDER_RE = re.compile(r"__TAG_(?!\d+__)")
LABEL_PREFIX_RE = re.compile(
    r"^(?P<head>\s*(?:__TAG_\d+__\s*)*)(?P<label>ASS|SRT|SUBS?|CAPTION|DIALOGUE)\s*(?::|-)\s*",
    flags=re.IGNORECASE,
)
SPANISH_MARKER_RE = re.compile(r"[áéíóúñüÁÉÍÓÚÑÜ]")
CREDITS_OVERRIDE_RE = re.compile(r"\bBrought to you by\s*\[[^\]]+\]", flags=re.IGNORECASE)
HONORIFICS = {"mr", "mrs", "ms", "miss", "sir", "ma'am", "madam"}
SUMMARY_MARKERS = (
    "summary",
    "resumen",
    "en esta escena",
    "en este episodio",
    "a lo largo de",
)
EN_STOPWORDS = {
    "the", "and", "you", "your", "for", "with", "that", "this", "was", "are",
    "but", "not", "from", "they", "she", "him", "her", "have", "has", "had",
    "what", "who", "where", "when", "why", "how", "can", "could", "would",
    "should", "will", "just", "about", "into", "like", "there", "them", "then",
}
ES_STOPWORDS = {
    "el", "la", "los", "las", "y", "o", "de", "del", "en", "que", "por", "para",
    "con", "sin", "sobre", "este", "esta", "estos", "estas", "eso", "esa",
    "pero", "porque", "como", "cuando", "donde", "quien", "quienes", "ser",
    "soy", "eres", "somos", "son", "estoy", "estas", "esta", "estan",
}
BULK_DIR_NAME = "SUBS_BULK"
SINGLE_TRANSLATION_CACHE = {}
REPAIR_BATCH_MAX = 8
MAX_REPAIR_FALLBACK_LINES = 3
ASS_MAX_BATCH_ITEMS = 32
ONE_SHOT_MAX_BATCH_ITEMS = 64
DEFAULT_CTX_TOKENS = 4096
DEFAULT_PREDICT_TOKENS = 256
PROMPT_OVERHEAD_TOKENS = 640   # margen conservador para instrucciones/contexto/JSON
DEFAULT_OUT_FRACTION = 0.60    # si num_predict no viene definido, reservar ~60% ctx para output
TRANSLATION_SUMMARY_MAX = 600
TRANSLATION_TONE_MAX = 400
ROLLING_CONTEXT_MAX_CHARS = 600
T = TypeVar("T")
FORMAT_MODE = "auto"
MINIFY_JSON_PROMPTS = True
BENCH_MODE = False
AUTO_JSON_DISABLED = False
AUTO_JSON_ATTEMPTS = 0
AUTO_JSON_FAILS = 0


class SimpleConsole:
    def print(self, *args, **kwargs) -> None:
        print(*args)


class DummyProgress:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def add_task(self, description: str, total: int = 0):
        return 0

    def advance(self, task_id: int, advance: int = 1):
        return None


def get_console():
    if RICH_AVAILABLE:
        return Console()
    return SimpleConsole()


def cprint(console, text: str, style: str | None = None) -> None:
    if RICH_AVAILABLE and style:
        console.print(f"[{style}]{text}[/]")
    else:
        console.print(text)


def progress_bar(console):
    if RICH_AVAILABLE:
        return Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
    return DummyProgress()


class RuntimeMetrics:
    def __init__(self) -> None:
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)
        self.counters = defaultdict(int)

    def reset(self) -> None:
        self.seconds.clear()
        self.calls.clear()
        self.counters.clear()

    @contextmanager
    def timed(self, key: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[key] += time.perf_counter() - start
            self.calls[key] += 1

    def bump(self, key: str, amount: int = 1) -> None:
        self.counters[key] += amount

    def total_by_prefix(self, prefix: str) -> float:
        return sum(value for name, value in self.seconds.items() if name.startswith(prefix))


RUNTIME_METRICS = RuntimeMetrics()


def set_runtime_flags(format_mode: str, minify_json: bool, bench: bool) -> None:
    global FORMAT_MODE, MINIFY_JSON_PROMPTS, BENCH_MODE
    global AUTO_JSON_DISABLED, AUTO_JSON_ATTEMPTS, AUTO_JSON_FAILS
    mode = (format_mode or "auto").strip().lower()
    if mode not in {"auto", "json", "schema"}:
        mode = "auto"
    FORMAT_MODE = mode
    MINIFY_JSON_PROMPTS = bool(minify_json)
    BENCH_MODE = bool(bench)
    AUTO_JSON_DISABLED = False
    AUTO_JSON_ATTEMPTS = 0
    AUTO_JSON_FAILS = 0


def resolve_format_mode(mode: str | None = None) -> str:
    selected = (mode or FORMAT_MODE or "auto").strip().lower()
    if selected not in {"auto", "json", "schema"}:
        return "auto"
    return selected


def dump_json(data: Any, *, minify: bool | None = None) -> str:
    use_minify = MINIFY_JSON_PROMPTS if minify is None else bool(minify)
    if use_minify:
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return json.dumps(data, ensure_ascii=False)


def print_runtime_breakdown(console, total_elapsed: float) -> None:
    summary_elapsed = RUNTIME_METRICS.seconds.get("stage.summary", 0.0)
    tone_elapsed = RUNTIME_METRICS.seconds.get("stage.tone_guide", 0.0)
    translate_elapsed = RUNTIME_METRICS.seconds.get("stage.translate", 0.0)
    retry_chat_elapsed = RUNTIME_METRICS.total_by_prefix("retry.chat.")
    translate_core_elapsed = max(0.0, translate_elapsed - retry_chat_elapsed)
    other_elapsed = max(0.0, total_elapsed - (summary_elapsed + tone_elapsed + translate_elapsed))

    cprint(console, "Timing breakdown:", "bold cyan")
    cprint(console, f"- summary: {summary_elapsed:.1f}s", "cyan")
    cprint(console, f"- tone guide: {tone_elapsed:.1f}s", "cyan")
    cprint(console, f"- translate core: {translate_core_elapsed:.1f}s", "cyan")
    cprint(console, f"- retries (LLM): {retry_chat_elapsed:.1f}s", "cyan")
    if other_elapsed >= 0.1:
        cprint(console, f"- other: {other_elapsed:.1f}s", "cyan")

    retry_attempts = (
        RUNTIME_METRICS.counters.get("retry.batch_temp0.attempts", 0)
        + RUNTIME_METRICS.counters.get("retry.srt_linewise.attempts", 0)
        + RUNTIME_METRICS.counters.get("retry.single_item.attempts", 0)
        + RUNTIME_METRICS.counters.get("retry.ass_surgical.attempts", 0)
        + RUNTIME_METRICS.counters.get("retry.repair_batch.attempts", 0)
    )
    if retry_attempts:
        cprint(
            console,
            (
                "Retry counters: "
                f"temp0_batches={RUNTIME_METRICS.counters.get('retry.batch_temp0.attempts', 0)} "
                f"(items={RUNTIME_METRICS.counters.get('retry.batch_temp0.items', 0)}), "
                f"srt_linewise={RUNTIME_METRICS.counters.get('retry.srt_linewise.attempts', 0)} "
                f"(lines={RUNTIME_METRICS.counters.get('retry.srt_linewise.lines', 0)}), "
                f"single={RUNTIME_METRICS.counters.get('retry.single_item.attempts', 0)}, "
                f"ass_surgical={RUNTIME_METRICS.counters.get('retry.ass_surgical.attempts', 0)} "
                f"(items={RUNTIME_METRICS.counters.get('retry.ass_surgical.items', 0)}), "
                f"repair_batch={RUNTIME_METRICS.counters.get('retry.repair_batch.attempts', 0)} "
                f"(items={RUNTIME_METRICS.counters.get('retry.repair_batch.items', 0)})"
            ),
            "cyan",
        )
    batch_calls = RUNTIME_METRICS.counters.get("translate.json_batch.calls", 0)
    top_batches = RUNTIME_METRICS.counters.get("translate.top_level_batches", 0)
    if top_batches:
        top_items = RUNTIME_METRICS.counters.get("translate.top_level_batch_items", 0)
        cprint(
            console,
            f"Top-level batches: {top_batches} (avg_items={top_items / float(top_batches):.1f})",
            "cyan",
        )
    if batch_calls:
        batch_items = RUNTIME_METRICS.counters.get("translate.json_batch.items", 0)
        batch_chars = RUNTIME_METRICS.counters.get("translate.json_batch.input_chars", 0)
        avg_items = batch_items / float(batch_calls)
        avg_chars = batch_chars / float(batch_calls)
        cprint(
            console,
            f"Batch stats: calls={batch_calls}, avg_items={avg_items:.1f}, avg_input_chars={avg_chars:.0f}",
            "cyan",
        )
    schema_retries = RUNTIME_METRICS.counters.get("format.schema_retry.attempts", 0)
    if schema_retries:
        cprint(console, f"Format fallbacks: schema_retry={schema_retries}", "cyan")
    if resolve_format_mode() == "auto":
        cprint(
            console,
            (
                f"Auto-format stats: json_attempts={AUTO_JSON_ATTEMPTS}, "
                f"json_fails={AUTO_JSON_FAILS}, json_disabled={AUTO_JSON_DISABLED}"
            ),
            "cyan",
        )
    ollama_calls = RUNTIME_METRICS.counters.get("ollama.calls", 0)
    if ollama_calls:
        avg_ollama = RUNTIME_METRICS.seconds.get("ollama.chat", 0.0) / float(ollama_calls)
        avg_prompt_chars = RUNTIME_METRICS.counters.get("ollama.prompt_chars", 0) / float(ollama_calls)
        cprint(
            console,
            f"Ollama calls: {ollama_calls} (avg={avg_ollama:.2f}s, avg_prompt_chars={avg_prompt_chars:.0f})",
            "cyan",
        )


def read_text(path: Path) -> Tuple[str, str, bool, bool]:
    data = path.read_bytes()
    has_bom = data.startswith(b"\xef\xbb\xbf")
    text = None
    last_err = None
    for enc in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            text = data.decode(enc)
            break
        except UnicodeDecodeError as exc:
            last_err = exc
    if text is None:
        raise RuntimeError(f"Failed to decode {path} ({last_err})")
    line_ending = "\r\n" if b"\r\n" in data else "\n"
    final_newline = data.endswith(b"\n")
    return text, line_ending, final_newline, has_bom


def write_text(path: Path, lines: List[str], line_ending: str, final_newline: bool, bom: bool) -> None:
    text = line_ending.join(lines)
    if final_newline:
        text += line_ending
    data = text.encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)

class OllamaClient:
    def __init__(self, host: str, model: str, timeout: int, keep_alive: str | None = "10m") -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.keep_alive = keep_alive

    def chat(self, messages, options=None, format=None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.keep_alive:
            payload["keep_alive"] = self.keep_alive
        if options:
            payload["options"] = options
        if format is not None:
            payload["format"] = format
        url = f"{self.host}/api/chat"
        data = dump_json(payload, minify=True).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
        prompt_chars = sum(len(str(msg.get("content", ""))) for msg in (messages or []))
        format_name = "none"
        if isinstance(format, str):
            format_name = format
        elif isinstance(format, dict):
            format_name = "schema"
        started = time.perf_counter()
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except socket.timeout as exc:
            raise OllamaTimeoutError(f"Ollama request timed out after {self.timeout}s") from exc
        except TimeoutError as exc:
            raise OllamaTimeoutError(f"Ollama request timed out after {self.timeout}s") from exc
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise OllamaTimeoutError(f"Ollama request timed out after {self.timeout}s") from exc
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Is it running?"
            ) from exc
        elapsed = time.perf_counter() - started
        RUNTIME_METRICS.seconds["ollama.chat"] += elapsed
        RUNTIME_METRICS.calls["ollama.chat"] += 1
        RUNTIME_METRICS.bump("ollama.calls")
        RUNTIME_METRICS.bump("ollama.prompt_chars", prompt_chars)
        RUNTIME_METRICS.bump("ollama.request_bytes", len(data))
        if BENCH_MODE:
            print(
                f"[bench] ollama chat {elapsed:.2f}s format={format_name} "
                f"prompt_chars={prompt_chars} request_bytes={len(data)}"
            )
        try:
            payload = json.loads(body)
            content = payload["message"]["content"]
            RUNTIME_METRICS.bump("ollama.response_chars", len(content))
            if BENCH_MODE:
                print(f"[bench] ollama out_chars={len(content)}")
            return content
        except Exception as exc:
            raise RuntimeError(f"Unexpected Ollama response: {body[:200]}") from exc


class OllamaTimeoutError(RuntimeError):
    pass


def build_chunks(lines: List[str], max_chars: int) -> List[str]:
    chunks = []
    cur = []
    cur_len = 0
    for line in lines:
        if not line.strip():
            continue
        line_len = len(line) + 1
        if cur and cur_len + line_len > max_chars:
            chunks.append("\n".join(cur))
            cur = [line]
            cur_len = line_len
        else:
            cur.append(line)
            cur_len += line_len
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def summarize_subs(
    client: OllamaClient,
    lines: List[str],
    max_chars: int,
    options,
    console,
) -> str:
    if not lines:
        return ""
    with RUNTIME_METRICS.timed("summary.total"):
        chunks = build_chunks(lines, max_chars)
        RUNTIME_METRICS.bump("summary.chunks", len(chunks))
        summaries = []
        system_msg = (
            "You summarize subtitles. Write a concise summary in Spanish. "
            "Keep it short and factual."
        )
        with progress_bar(console) as progress:
            task_id = progress.add_task("Summary", total=len(chunks))
            for chunk in chunks:
                user_msg = (
                    "Summarize the following subtitle text in Spanish. "
                    "Keep it short (6-8 sentences max).\n\n" + chunk
                )
                with RUNTIME_METRICS.timed("summary.chat.chunk"):
                    content = client.chat(
                        [
                            {"role": "system", "content": system_msg},
                            {"role": "user", "content": user_msg},
                        ],
                        options=options,
                    )
                summaries.append(content.strip())
                progress.advance(task_id, 1)
        if len(summaries) == 1:
            return summaries[0]
        user_msg = (
            "Combine these partial summaries into a single short summary in Spanish. "
            "Avoid repetition.\n\n" + "\n\n".join(summaries)
        )
        with RUNTIME_METRICS.timed("summary.chat.merge"):
            return client.chat(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options=options,
            ).strip()


def build_tone_guide(
    client: OllamaClient,
    summary: str,
    sample_lines: List[str],
    options,
    console,
) -> str:
    if not summary:
        return ""
    sample = "\n".join(sample_lines[:80])
    system_msg = (
        "You analyze subtitle tone and propose a short translation style guide in Spanish."
    )
    user_msg = (
        "Based on the summary and sample dialogue, output a compact JSON object with keys: "
        "overall_tone (string), register_rules (array of short rules), sarcasm (array), "
        "double_meaning (array), addressing (array). Keep it brief. JSON only.\n\n"
        f"Summary: {summary}\n\n"
        f"Sample lines:\n{sample}"
    )
    cprint(console, "Building tone guide...", "bold cyan")
    with RUNTIME_METRICS.timed("tone_guide.total"):
        with RUNTIME_METRICS.timed("tone_guide.chat"):
            content = client.chat(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options=options,
            )
    data = extract_json_object(content)
    if not data:
        guide = content.strip()
        return guide[:1200]

    def _join_list(value):
        if isinstance(value, list):
            return "; ".join(str(v) for v in value if str(v).strip())
        if value is None:
            return ""
        return str(value)

    parts = []
    overall = _join_list(data.get("overall_tone"))
    if overall:
        parts.append(f"Tono: {overall}")
    reg = _join_list(data.get("register_rules"))
    if reg:
        parts.append(f"Registro: {reg}")
    sar = _join_list(data.get("sarcasm"))
    if sar:
        parts.append(f"Sarcasmo: {sar}")
    dbl = _join_list(data.get("double_meaning"))
    if dbl:
        parts.append(f"Doble sentido: {dbl}")
    addr = _join_list(data.get("addressing"))
    if addr:
        parts.append(f"Tratamiento: {addr}")
    guide = " | ".join(parts)
    return guide[:1200]


def extract_json_array(text: str) -> List[str] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
        if isinstance(data, list):
            return [str(item) for item in data]
    except json.JSONDecodeError:
        return None
    return None


def extract_json_array_of_arrays(text: str) -> List[List[str]] | None:
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("[")
    end = cleaned.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    out: List[List[str]] = []
    for row in data:
        if not isinstance(row, list):
            return None
        out.append([str(item) for item in row])
    return out


def extract_json_object(text: str) -> dict | None:
    cleaned = text.strip()
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "")
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        data = json.loads(candidate)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return None
    return None


def count_words(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9']+", text))


def ascii_word_tokens(text: str) -> List[str]:
    return [t.lower() for t in re.findall(r"[A-Za-z']+", text)]


def normalize_target_lang(target_lang: str) -> str:
    lowered = target_lang.strip().lower()
    if lowered in {"spanish", "es", "es-419", "es_419", "espanol"} or lowered.startswith("spanish"):
        return "es"
    if lowered in {"english", "en", "en-us", "en_us", "en-gb", "en_gb"} or lowered.startswith("english"):
        return "en"
    return lowered


def has_label_prefix_artifact(text: str) -> bool:
    return LABEL_PREFIX_RE.match(text or "") is not None


def strip_label_prefix(text: str) -> str:
    cleaned = text or ""
    # Some models prepend labels repeatedly (e.g., "ASS: SRT: ...").
    for _ in range(3):
        match = LABEL_PREFIX_RE.match(cleaned)
        if not match:
            break
        cleaned = f"{match.group('head')}{cleaned[match.end():]}"
    return cleaned


def has_target_language_leak(source: str, translated: str, target_lang: str) -> bool:
    mode = normalize_target_lang(target_lang)
    if mode not in {"es", "en"}:
        return False
    dst_tokens = ascii_word_tokens(translated)
    if not dst_tokens:
        return False

    en_hits = sum(1 for token in dst_tokens if token in EN_STOPWORDS)
    es_hits = sum(1 for token in dst_tokens if token in ES_STOPWORDS)
    has_spanish_markers = bool(SPANISH_MARKER_RE.search(translated))
    has_spanish_signal = has_spanish_markers or es_hits >= 2
    has_english_signal = en_hits >= 2 or "i" in dst_tokens

    if mode == "es":
        # A standalone "I" in Spanish output is a strong incomplete-translation signal.
        if "i" in dst_tokens:
            return True
        if en_hits >= 2 and not has_spanish_signal:
            return True
        return False

    # Symmetric check when target is English.
    if es_hits >= 2 and not has_english_signal:
        return True
    if has_spanish_markers and not has_english_signal:
        src_tokens = set(ascii_word_tokens(source))
        if not src_tokens.intersection({"senor", "senora", "senorita", "espanol"}):
            return True
    return False


def is_unchanged_english(source_plain: str, translated_plain: str, target_lang: str) -> bool:
    if normalize_target_lang(target_lang) != "es":
        return False
    src = (source_plain or "").strip()
    dst = (translated_plain or "").strip()
    if not src or not dst:
        return False
    if src.lower() != dst.lower():
        return False
    tokens = ascii_word_tokens(src)
    if not tokens:
        return False
    if any(token in EN_STOPWORDS for token in tokens):
        return True
    if any(token in HONORIFICS for token in tokens):
        return True
    if len(tokens) >= 3:
        has_spanish_markers = bool(SPANISH_MARKER_RE.search(dst))
        es_hits = sum(1 for token in ascii_word_tokens(dst) if token in ES_STOPWORDS)
        if not has_spanish_markers and es_hits == 0:
            return True
    return False


def apply_phrase_overrides(text: str) -> str:
    return CREDITS_OVERRIDE_RE.sub("Tra\u00eddo por [el_inmortus]", text or "")


def is_very_suspicious_translation(source: str, translated: str) -> bool:
    src = source.strip()
    dst = translated.strip()
    if not dst:
        return True
    src_words = count_words(src)
    dst_words = count_words(dst)
    src_len = len(src)
    dst_len = len(dst)
    dst_lower = dst.lower()

    if src_words <= 4 and dst_words >= 35:
        return True
    if dst_words > max(60, src_words * 6):
        return True
    if dst_len > max(420, src_len * 6 + 120):
        return True
    if dst_words >= 25 and any(marker in dst_lower for marker in SUMMARY_MARKERS):
        return True
    return False


def is_suspicious_translation(source: str, translated: str) -> bool:
    src = source.strip()
    dst = translated.strip()
    if not dst:
        return True

    src_words = count_words(src)
    dst_words = count_words(dst)
    src_len = len(src)
    dst_len = len(dst)
    src_lower = src.lower()
    dst_lower = dst.lower()

    if src_lower == dst_lower:
        src_ascii = ascii_word_tokens(src)
        if any(token in EN_STOPWORDS for token in src_ascii):
            return True

    if src_words <= 2:
        return dst_words >= 18 and dst_len > 90
    if src_words <= 12 and dst_words > max(30, src_words * 4):
        return True
    if src_len <= 80 and dst_len > 220:
        return True
    if dst_len > max(320, int(src_len * 4) + 120):
        return True

    if src_len >= 20 and src_lower in dst_lower and dst_len > src_len + 12:
        return True

    src_ascii = ascii_word_tokens(src)
    dst_ascii = set(ascii_word_tokens(dst))
    if src_ascii and dst_ascii:
        contraction_tokens = [t for t in src_ascii if "'" in t]
        if any(t in dst_ascii for t in contraction_tokens):
            return True
        shared_long = [
            t
            for t in src_ascii
            if len(t) >= 4 and t in dst_ascii and t not in {"tsuda", "ito", "nadeshiko", "toonshub"}
        ]
        if len(set(shared_long)) >= 2:
            return True

    if dst_words > 35 and any(marker in dst_lower for marker in SUMMARY_MARKERS):
        return True
    return False


def ass_placeholders_match(source_protected: str, translated: str) -> bool:
    expected = PLACEHOLDER_TOKEN_RE.findall(source_protected)
    actual = PLACEHOLDER_TOKEN_RE.findall(translated)
    return expected == actual


def has_ass_markup(text: str) -> bool:
    return ASS_TAG_RE.search(text) is not None


def ass_structure_preserved(source: str, translated: str) -> bool:
    return ASS_TAG_RE.findall(source) == ASS_TAG_RE.findall(translated)


def has_placeholder_artifacts(text: str) -> bool:
    if not text:
        return False
    if LEGACY_PLACEHOLDER_RE.search(text):
        return True
    return BROKEN_PLACEHOLDER_RE.search(text) is not None


def guarded_translate_one(
    client: OllamaClient,
    text: str,
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
    source_for_checks: str | None = None,
    validator=None,
    prefer_no_context: bool = False,
    chat_timing_label: str = "translate.chat.single",
) -> str | None:
    source_check = source_for_checks if source_for_checks is not None else text
    if prefer_no_context:
        attempts = [("", "")]
    else:
        attempts = [(summary, tone_guide), ("", "")]
        if not summary and not tone_guide:
            attempts = [("", "")]
    for sum_ctx, tone_ctx in attempts:
        cache_key = (
            text,
            sum_ctx,
            tone_ctx,
            target_lang,
            tuple(sorted((options or {}).items())),
        )
        candidate = SINGLE_TRANSLATION_CACHE.get(cache_key)
        if candidate is None:
            candidate = translate_one(
                client,
                text,
                sum_ctx,
                tone_ctx,
                target_lang,
                options,
                chat_timing_label=chat_timing_label,
            )
            SINGLE_TRANSLATION_CACHE[cache_key] = candidate
        if not candidate:
            continue
        candidate = strip_label_prefix(candidate)
        if has_label_prefix_artifact(candidate):
            continue
        if has_placeholder_artifacts(candidate) and "__TAG_" not in text:
            continue
        if not ass_placeholders_match(text, candidate):
            continue
        if validator and not validator(candidate):
            continue
        if has_target_language_leak(source_check, candidate, target_lang):
            continue
        if is_suspicious_translation(source_check, candidate):
            continue
        return candidate
    return None


def translate_ass_text_field_segmented(
    client: OllamaClient,
    text_field: str,
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
) -> str:
    tokens = split_ass_text(text_field)
    segments: List[Tuple[int, str]] = []
    for idx, token in enumerate(tokens):
        if token.startswith("{") or token in ("\\N", "\\n", "\\h"):
            continue
        if not should_translate(token):
            continue
        segments.append((idx, token))

    if not segments:
        return "".join(tokens)

    sources = [segment for _, segment in segments]
    batch_out = translate_batch(client, sources, summary, tone_guide, target_lang, options)
    if batch_out is None or len(batch_out) != len(sources):
        batch_out = [None] * len(sources)

    for (idx, source), candidate in zip(segments, batch_out):
        fixed = None
        if candidate:
            candidate = strip_label_prefix(candidate)
            if (
                not has_ass_markup(candidate)
                and not has_placeholder_artifacts(candidate)
                and not has_label_prefix_artifact(candidate)
                and not has_target_language_leak(source, candidate, target_lang)
                and not is_suspicious_translation(source, candidate)
            ):
                fixed = candidate

        safe = guarded_translate_one(
            client,
            source,
            summary,
            tone_guide,
            target_lang,
            options,
            source_for_checks=source,
            validator=lambda cand, src=source: not has_ass_markup(cand)
            and not has_placeholder_artifacts(cand)
            and not has_label_prefix_artifact(cand)
            and not has_target_language_leak(src, cand, target_lang),
            prefer_no_context=True,
        ) if fixed is None else fixed
        tokens[idx] = safe if safe is not None else source
    return "".join(tokens)


def translate_batch(
    client: OllamaClient,
    texts: List[str],
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
) -> List[str] | None:
    summary_short = summary.strip()
    if len(summary_short) > 1200:
        summary_short = summary_short[:1200] + "..."
    tone_short = tone_guide.strip()
    if len(tone_short) > 800:
        tone_short = tone_short[:800] + "..."
    has_placeholders = any("__TAG_" in item for item in texts)
    placeholder_rule = (
        "If placeholders like __TAG_0__ appear, keep them exactly. "
        if has_placeholders
        else "Do not introduce placeholders like __TAG_0__ if they are not in the source. "
    )
    system_msg = (
        "You are a professional subtitle translator. "
        "Translate from English to the target language. "
        "Preserve meaning, tone, names, and punctuation. "
        "Do not add explanations."
    )
    user_msg = (
        f"Context summary (for consistency): {summary_short}\n"
        f"Tone guide (apply per line): {tone_short}\n"
        f"Translate each item in the JSON array from English to {target_lang}. "
        "Keep line breaks. Keep ASS tags in braces and control codes \\N, \\n, \\h unchanged. "
        "Never prepend labels like ASS:, SRT:, SUB:, or Dialogue:. "
        "Translate only the current line, never summarize a scene or multiple lines. "
        "Fully translate every word from the source; do not drop connectors or leave source fragments. "
        f"{placeholder_rule}"
        "Respect sarcasm, double meanings, and register (formal/informal) as guided; do not neutralize tone. "
        "Return ONLY a JSON array of strings, same length and order, in compact/minified JSON "
        "(single line, no extra whitespace/newlines).\n\n"
        f"Input JSON: {dump_json(texts)}"
    )
    with RUNTIME_METRICS.timed("translate.chat.batch_legacy"):
        content = client.chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            options=options,
        )
    return extract_json_array(content)


def translate_one(
    client: OllamaClient,
    text: str,
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
    chat_timing_label: str = "translate.chat.single",
) -> str:
    summary_short = summary.strip()
    if len(summary_short) > 1200:
        summary_short = summary_short[:1200] + "..."
    tone_short = tone_guide.strip()
    if len(tone_short) > 800:
        tone_short = tone_short[:800] + "..."
    has_placeholders = "__TAG_" in text
    placeholder_rule = (
        "If placeholders like __TAG_0__ appear, keep them exactly.\n\n"
        if has_placeholders
        else "Do not introduce placeholders like __TAG_0__ if they are not in the source.\n\n"
    )
    system_msg = (
        "You are a professional subtitle translator. "
        "Translate from English to the target language. "
        "Preserve meaning, tone, names, and punctuation. "
        "Do not add explanations."
    )
    user_msg = (
        f"Context summary (for consistency): {summary_short}\n"
        f"Tone guide (apply per line): {tone_short}\n"
        f"Translate to {target_lang}. Return ONLY the translated text. "
        "Keep ASS tags in braces and control codes \\N, \\n, \\h unchanged. "
        "Never prepend labels like ASS:, SRT:, SUB:, or Dialogue:. "
        "Translate only this line, never summarize the plot or add narration. "
        "Fully translate every word from the source; do not drop connectors or leave source fragments. "
        f"{placeholder_rule}"
        "Respect sarcasm, double meanings, and register (formal/informal) as guided; do not neutralize tone.\n\n"
        f"Text: {text}"
    )
    with RUNTIME_METRICS.timed(chat_timing_label):
        return client.chat(
            [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            options=options,
        ).strip()


def repair_batch(
    client: OllamaClient,
    items: List[dict],
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
) -> List[str | None]:
    if not items:
        return []
    RUNTIME_METRICS.bump("retry.repair_batch.attempts")
    RUNTIME_METRICS.bump("retry.repair_batch.items", len(items))
    include_context = len(items) <= 3
    summary_short = summary.strip()[:240] if include_context and summary else ""
    tone_short = tone_guide.strip()[:160] if include_context and tone_guide else ""

    def build_messages(chunk_items: List[dict]):
        payload_items = []
        for item in chunk_items:
            payload_items.append(
                {
                    "src": str(item.get("src", "")),
                    "bad": str(item.get("bad", "")),
                }
            )
        user_msg = (
            f"Fix each `bad` subtitle so it is correct {target_lang}.\n"
            "Rules:\n"
            "- Never prepend labels like ASS:, SRT:, SUB:, SUBS:, CAPTION:, Dialogue:.\n"
            "- Fully translate remaining English fragments.\n"
            "- Keep placeholders __TAG_n__ exactly as in src (same count/order).\n"
            "- Do not add ASS tags/braces/control codes not present in src.\n"
            "- Output ONLY JSON array of strings, same order/length, as compact/minified JSON.\n"
        )
        if summary_short:
            user_msg += f"Summary hint: {summary_short}\n"
        if tone_short:
            user_msg += f"Tone hint: {tone_short}\n"
        user_msg += f"Input JSON: {dump_json(payload_items)}"
        return [
            {"role": "system", "content": "You are a subtitle translation fixer. Output ONLY compact/minified JSON."},
            {"role": "user", "content": user_msg},
        ]

    def build_repair_options(chunk_len: int):
        repair_options = dict(options or {})
        max_predict = min(512, max(128, chunk_len * 128))
        if repair_options.get("num_predict") is None or int(repair_options["num_predict"]) > max_predict:
            repair_options["num_predict"] = max_predict
        return repair_options

    def repair_chunk_with_split(chunk_items: List[dict]) -> List[str | None]:
        if not chunk_items:
            return []
        try:
            with RUNTIME_METRICS.timed("retry.chat.repair_batch"):
                content = client.chat(
                    build_messages(chunk_items),
                    options=build_repair_options(len(chunk_items)),
                )
        except OllamaTimeoutError:
            if len(chunk_items) == 1:
                return [None]
            mid = max(1, len(chunk_items) // 2)
            return repair_chunk_with_split(chunk_items[:mid]) + repair_chunk_with_split(chunk_items[mid:])
        except RuntimeError as exc:
            msg = str(exc).lower()
            if "timed out" in msg and len(chunk_items) > 1:
                mid = max(1, len(chunk_items) // 2)
                return repair_chunk_with_split(chunk_items[:mid]) + repair_chunk_with_split(chunk_items[mid:])
            return [None] * len(chunk_items)
        out = extract_json_array(content)
        if out is None or len(out) != len(chunk_items):
            return [None] * len(chunk_items)
        return [strip_label_prefix(item) for item in out]

    fixed: List[str | None] = []
    for chunk in batched(items, REPAIR_BATCH_MAX):
        fixed.extend(repair_chunk_with_split(chunk))
    if len(fixed) != len(items):
        return [None] * len(items)
    return fixed


def batched(items: List[T], size: int) -> List[List[T]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def effective_batch_cap(batch_size: int, one_shot: bool) -> int:
    requested = max(1, int(batch_size))
    if one_shot:
        return min(requested, ONE_SHOT_MAX_BATCH_ITEMS)
    return min(requested, ASS_MAX_BATCH_ITEMS)


def should_translate(segment: str) -> bool:
    return bool(re.search(r"[A-Za-z]", segment))


def split_ass_text(text: str) -> List[str]:
    parts = ASS_TAG_RE.split(text)
    return [p for p in parts if p is not None and p != ""]


def ass_protect_tags(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    replacements: List[Tuple[str, str]] = []

    def repl(match: re.Match) -> str:
        key = f"__TAG_{len(replacements)}__"
        replacements.append((key, match.group(0)))
        return key

    protected = ASS_TAG_RE.sub(repl, text)
    return protected, replacements


def ass_restore_tags(text: str, replacements: List[Tuple[str, str]]) -> str:
    restored = text
    for key, value in replacements:
        restored = restored.replace(key, value)
    return restored


def ass_plain_text(text: str) -> str:
    text = ASS_STRIP_RE.sub("", text)
    text = text.replace("\\N", "\n").replace("\\n", "\n").replace("\\h", " ")
    return text


def trim_translation_context(summary: str, tone_guide: str) -> Tuple[str, str]:
    summary_short = (summary or "").strip()
    tone_short = (tone_guide or "").strip()
    if len(summary_short) > TRANSLATION_SUMMARY_MAX:
        summary_short = summary_short[:TRANSLATION_SUMMARY_MAX].rstrip() + "..."
    if len(tone_short) > TRANSLATION_TONE_MAX:
        tone_short = tone_short[:TRANSLATION_TONE_MAX].rstrip() + "..."
    return summary_short, tone_short


def estimate_tokens_from_text(text: str) -> int:
    return max(1, int(math.ceil(len(text or "") / 4.0)))


def translation_budget(options) -> Tuple[int, int]:
    opts = options or {}
    num_ctx = int(opts.get("num_ctx") or DEFAULT_CTX_TOKENS)
    num_predict = int(opts.get("num_predict") or 0)
    # Reserva de salida coherente con cómo luego calculas num_predict dinámico.
    reserve_out = num_predict if num_predict > 0 else int(math.floor(num_ctx * DEFAULT_OUT_FRACTION))
    reserve_out = max(256, min(4096, reserve_out))
    # Presupuesto de entrada real: ctx - reserva_out - overhead fijo.
    budget = max(256, num_ctx - reserve_out - PROMPT_OVERHEAD_TOKENS)
    return budget, reserve_out


def build_fixed_array_schema(size: int) -> dict:
    return {
        "type": "array",
        "items": {"type": "string"},
        "minItems": size,
        "maxItems": size,
    }


def build_fixed_array_of_arrays_schema(size: int) -> dict:
    return {
        "type": "array",
        "items": {
            "type": "array",
            "items": {"type": "string"},
        },
        "minItems": size,
        "maxItems": size,
    }


def parse_array_response(content: str, expected_len: int) -> List[str] | None:
    cleaned = content.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) == expected_len:
            return [str(item) for item in data]
        if expected_len == 1:
            if isinstance(data, str):
                return [data]
            if isinstance(data, dict):
                for key in ("translation", "translated", "text", "value", "output", "result"):
                    value = data.get(key)
                    if isinstance(value, str):
                        return [value]
                if len(data) == 1:
                    only_value = next(iter(data.values()))
                    if isinstance(only_value, str):
                        return [only_value]
    except json.JSONDecodeError:
        pass
    fallback = extract_json_array(content)
    if fallback is not None and len(fallback) == expected_len:
        return fallback
    return None


def parse_array_of_arrays_response(content: str, expected_len: int) -> List[List[str]] | None:
    cleaned = content.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list) and len(data) == expected_len and all(isinstance(row, list) for row in data):
            return [[str(item) for item in row] for row in data]
    except json.JSONDecodeError:
        pass
    fallback = extract_json_array_of_arrays(content)
    if fallback is not None and len(fallback) == expected_len:
        return fallback
    return None


def build_options_for_batch(options, texts: List[str], force_temperature_zero: bool = False) -> dict:
    options_batch = dict(options or {})
    num_ctx = int(options_batch.get("num_ctx") or DEFAULT_CTX_TOKENS)
    in_tokens = sum(estimate_tokens_from_text(text) for text in texts)
    est_out = int(math.ceil(in_tokens * 1.2)) + 64
    base_predict = int(options_batch.get("num_predict") or 0)
    desired_predict = max(base_predict, est_out)
    # Cap por tokens realmente disponibles (evita truncar JSON a mitad).
    available = max(64, num_ctx - in_tokens - PROMPT_OVERHEAD_TOKENS)
    cap = min(4096, available)
    options_batch["num_predict"] = max(64, min(desired_predict, cap))
    if force_temperature_zero:
        options_batch["temperature"] = 0.0
    return options_batch


def rolling_context_snippet(history: List[str], size: int) -> str:
    if size <= 0 or not history:
        return ""
    snippet = "\n".join(line for line in history[-size:] if line.strip())
    if len(snippet) > ROLLING_CONTEXT_MAX_CHARS:
        snippet = snippet[-ROLLING_CONTEXT_MAX_CHARS:]
    return snippet


def build_adaptive_batches(
    items: List[dict],
    batch_size: int,
    options,
    one_shot: bool,
    console,
) -> List[List[dict]]:
    if not items:
        return []
    budget, reserve_out = translation_budget(options)
    cap = len(items) if batch_size is None or int(batch_size) <= 0 else max(1, int(batch_size))
    est_total = sum(item.get("in_tokens", estimate_tokens_from_text(item.get("protected_text", ""))) for item in items)
    if est_total + reserve_out <= budget and len(items) <= cap:
        return [items]

    if one_shot:
        if len(items) > cap:
            cprint(
                console,
                f"--one-shot requested but batch-size={cap} < items={len(items)}; using adaptive batches.",
                "yellow",
            )
        elif est_total + reserve_out <= budget:
            return [items]
        else:
            cprint(
                console,
                "--one-shot requested but estimated prompt does not fit num_ctx; using adaptive batches.",
                "yellow",
            )

    batches: List[List[dict]] = []
    cur: List[dict] = []
    cur_tokens = 0
    for item in items:
        in_tokens = item.get("in_tokens", estimate_tokens_from_text(item.get("protected_text", "")))
        if not cur:
            cur = [item]
            cur_tokens = in_tokens
            continue
        # `budget` already discounts reserved output tokens, so avoid double subtraction.
        fits_tokens = (cur_tokens + in_tokens) <= budget
        fits_count = len(cur) < cap
        if fits_tokens and fits_count:
            cur.append(item)
            cur_tokens += in_tokens
            continue
        batches.append(cur)
        cur = [item]
        cur_tokens = in_tokens
    if cur:
        batches.append(cur)
    return batches


def build_translation_messages(
    texts: List[str],
    target_lang: str,
    summary: str,
    tone_guide: str,
    mode: str,
    rolling_context: str = "",
) -> List[dict]:
    summary_short, tone_short = trim_translation_context(summary, tone_guide)
    context_lines = []
    if summary_short:
        context_lines.append(f"Summary hint: {summary_short}")
    if tone_short:
        context_lines.append(f"Tone hint: {tone_short}")
    if rolling_context:
        context_lines.append(f"Recent translated lines: {rolling_context}")
    context_blob = "\n".join(context_lines)

    if mode == "ass_protected":
        user_msg = (
            f"Translate each item from English to {target_lang}.\n"
            "Rules:\n"
            "1) Preserve placeholders like __TAG_0__ exactly (same count/order).\n"
            "2) Do NOT introduce labels such as ASS:, SRT:, Dialogue:.\n"
            "3) Do NOT add braces/tags/control codes. Translate only natural language.\n"
            "4) Do not translate content inside square brackets [like this]; preserve it exactly.\n"
            "5) Output ONLY a JSON array of strings, same length/order, as compact/minified JSON.\n"
        )
    else:
        user_msg = (
            f"Translate each subtitle block from English to {target_lang}.\n"
            "Rules:\n"
            "1) Keep line breaks in each item.\n"
            "2) Do NOT introduce labels such as ASS:, SRT:, Dialogue:.\n"
            "3) Do not translate content inside square brackets [like this]; preserve it exactly.\n"
            "4) Output ONLY a JSON array of strings, same length/order, as compact/minified JSON.\n"
        )

    if context_blob:
        user_msg += context_blob + "\n"
    user_msg += f"Input JSON: {dump_json(texts)}"
    return [
        {
            "role": "system",
            "content": "You are a professional subtitle translator. Output only valid compact/minified JSON.",
        },
        {"role": "user", "content": user_msg},
    ]


def translate_json_batch(
    client: OllamaClient,
    texts: List[str],
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
    mode: str,
    rolling_context: str = "",
    force_temperature_zero: bool = False,
    timing_label: str = "translate.chat.batch_json",
    format_mode: str | None = None,
) -> List[str] | None:
    global AUTO_JSON_DISABLED, AUTO_JSON_ATTEMPTS, AUTO_JSON_FAILS
    if not texts:
        return []
    input_json_chars = len(dump_json(texts))
    RUNTIME_METRICS.bump("translate.json_batch.calls")
    RUNTIME_METRICS.bump("translate.json_batch.items", len(texts))
    RUNTIME_METRICS.bump("translate.json_batch.input_chars", input_json_chars)
    selected_format_mode = resolve_format_mode(format_mode)
    if BENCH_MODE:
        print(
            f"[bench] batch items={len(texts)} input_json_chars={input_json_chars} "
            f"mode={mode} format_mode={selected_format_mode}"
        )
    messages = build_translation_messages(
        texts=texts,
        target_lang=target_lang,
        summary=summary,
        tone_guide=tone_guide,
        mode=mode,
        rolling_context=rolling_context,
    )
    schema = build_fixed_array_schema(len(texts))
    options_batch = build_options_for_batch(
        options=options,
        texts=texts,
        force_temperature_zero=force_temperature_zero,
    )
    if mode == "ass_protected" and not force_temperature_zero:
        options_batch["temperature"] = 0.0
    request_format: str | dict
    used_auto_json = False
    if selected_format_mode == "schema":
        request_format = schema
    elif selected_format_mode == "auto" and (
        force_temperature_zero or len(texts) <= 2 or AUTO_JSON_DISABLED or mode == "ass_protected"
    ):
        request_format = schema
    else:
        request_format = "json"
        used_auto_json = selected_format_mode == "auto"
    already_schema = isinstance(request_format, dict)
    try:
        already_temp0 = float(options_batch.get("temperature", 0.0)) == 0.0
    except (TypeError, ValueError):
        already_temp0 = False
    with RUNTIME_METRICS.timed(timing_label):
        content = client.chat(messages, options=options_batch, format=request_format)
    parsed = parse_array_response(content, len(texts))
    if parsed is not None:
        if used_auto_json:
            AUTO_JSON_ATTEMPTS += 1
        return parsed
    if used_auto_json:
        AUTO_JSON_ATTEMPTS += 1
        AUTO_JSON_FAILS += 1
        fail_rate = AUTO_JSON_FAILS / float(max(1, AUTO_JSON_ATTEMPTS))
        if AUTO_JSON_ATTEMPTS >= 10 and AUTO_JSON_FAILS >= 4 and fail_rate >= 0.25:
            AUTO_JSON_DISABLED = True
            if BENCH_MODE:
                print(
                    f"[bench] auto json disabled (attempts={AUTO_JSON_ATTEMPTS}, "
                    f"fails={AUTO_JSON_FAILS}, fail_rate={fail_rate:.2f})"
                )
    if selected_format_mode != "auto":
        return None
    if already_schema and already_temp0:
        return None
    # Fast path in auto mode: retry once with strict schema + temp=0 only on parse/length failure.
    RUNTIME_METRICS.bump("format.schema_retry.attempts")
    schema_options = build_options_for_batch(
        options=options,
        texts=texts,
        force_temperature_zero=True,
    )
    with RUNTIME_METRICS.timed(f"{timing_label}.schema_retry"):
        strict_content = client.chat(messages, options=schema_options, format=schema)
    return parse_array_response(strict_content, len(texts))


def translate_json_batch_with_split(
    client: OllamaClient,
    texts: List[str],
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
    mode: str,
    rolling_context: str = "",
    force_temperature_zero: bool = False,
    timing_label: str = "translate.chat.batch_json",
    format_mode: str | None = None,
) -> List[str | None]:
    if not texts:
        return []
    try:
        out = translate_json_batch(
            client,
            texts,
            summary,
            tone_guide,
            target_lang,
            options,
            mode=mode,
            rolling_context=rolling_context,
            force_temperature_zero=force_temperature_zero,
            timing_label=timing_label,
            format_mode=format_mode,
        )
    except OllamaTimeoutError:
        out = None
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            out = None
        else:
            out = None

    if out is not None and len(out) == len(texts):
        return [strip_label_prefix(item) for item in out]
    if len(texts) == 1:
        return [None]

    mid = max(1, len(texts) // 2)
    left = translate_json_batch_with_split(
        client,
        texts[:mid],
        summary,
        tone_guide,
        target_lang,
        options,
        mode=mode,
        rolling_context=rolling_context,
        force_temperature_zero=force_temperature_zero,
        timing_label=timing_label,
        format_mode=format_mode,
    )
    right = translate_json_batch_with_split(
        client,
        texts[mid:],
        summary,
        tone_guide,
        target_lang,
        options,
        mode=mode,
        rolling_context=rolling_context,
        force_temperature_zero=force_temperature_zero,
        timing_label=timing_label,
        format_mode=format_mode,
    )
    return left + right


def repair_issue_score(reasons: List[str]) -> int:
    weights = {
        "empty_output": 6,
        "placeholder_mismatch": 6,
        "unchanged": 6,
        "placeholder_artifacts": 5,
        "unexpected_ass_markup": 4,
        "language_leak": 4,
        "very_suspicious": 2,
    }
    return sum(weights.get(reason, 1) for reason in reasons)


def split_plain_text_for_slots(text: str, expected_slots: int) -> List[str]:
    if expected_slots <= 0:
        return []
    raw_lines = [segment.strip() for segment in text.split("\n")]
    if len(raw_lines) == expected_slots:
        return raw_lines
    merged = " ".join(segment for segment in raw_lines if segment).strip()
    if not merged:
        return [""] * expected_slots
    words = merged.split()
    if len(words) < expected_slots:
        words += [""] * (expected_slots - len(words))
    chunk = int(math.ceil(len(words) / float(expected_slots)))
    out = []
    cursor = 0
    for _ in range(expected_slots):
        out.append(" ".join(words[cursor : cursor + chunk]).strip())
        cursor += chunk
    return out


def recover_ass_candidate_placeholders(state: dict, candidate: str) -> str | None:
    cleaned = strip_label_prefix(candidate or "")
    if not cleaned.strip():
        return None
    plain = ass_plain_text(cleaned).strip()
    if not plain:
        return None
    tokens = list(state.get("tokens") or split_ass_text(state["text_field"]))
    slot_positions, _ = extract_ass_slots(tokens)
    if not slot_positions:
        return None
    translated_slots = split_plain_text_for_slots(plain, len(slot_positions))
    rebuilt = reinsert_ass_slots(tokens, slot_positions, translated_slots)
    rebuilt_protected, _ = ass_protect_tags(rebuilt)
    source_protected = state.get("protected") or state.get("protected_text")
    if source_protected and not ass_placeholders_match(source_protected, rebuilt_protected):
        return None
    return rebuilt_protected


def evaluate_ass_line_candidate(state: dict, cleaned: str, target_lang: str) -> dict:
    source_plain = ass_plain_text(state["text_field"])
    placeholder_ok = ass_placeholders_match(state["protected"], cleaned)
    markup_found = has_ass_markup(cleaned)
    placeholder_artifact_found = has_placeholder_artifacts(cleaned)
    restored = ass_restore_tags(cleaned, state["replacements"])
    restored_plain = ass_plain_text(restored)
    reasons = []
    if not cleaned.strip() or not restored_plain.strip():
        reasons.append("empty_output")
    if not placeholder_ok:
        reasons.append("placeholder_mismatch")
    if markup_found:
        reasons.append("unexpected_ass_markup")
    if placeholder_artifact_found or has_placeholder_artifacts(restored):
        reasons.append("placeholder_artifacts")
    if is_unchanged_english(source_plain, restored_plain, target_lang):
        reasons.append("unchanged")
    if has_target_language_leak(source_plain, restored_plain, target_lang):
        reasons.append("language_leak")
    if is_very_suspicious_translation(source_plain, restored_plain):
        reasons.append("very_suspicious")
    return {
        "candidate": cleaned,
        "restored": restored,
        "reasons": reasons,
    }


def scan_ass_line_candidate(state: dict, candidate: str, target_lang: str) -> dict:
    cleaned = strip_label_prefix(candidate or "")
    evaluation = evaluate_ass_line_candidate(state, cleaned, target_lang)
    reasons = evaluation["reasons"]

    # Recover common case where model translated text but dropped __TAG_n__ placeholders.
    if "placeholder_mismatch" in reasons:
        recovered = recover_ass_candidate_placeholders(state, cleaned)
        if recovered is not None and recovered != cleaned:
            evaluation = evaluate_ass_line_candidate(state, recovered, target_lang)
            reasons = evaluation["reasons"]

    ok = len(reasons) == 0
    needs_llm_repair = any(
        reason in {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged", "language_leak", "very_suspicious"}
        for reason in reasons
    )
    return {
        "ok": ok,
        "candidate": evaluation["candidate"],
        "restored": evaluation["restored"],
        "reasons": reasons,
        "needs_llm_repair": needs_llm_repair,
        "score": repair_issue_score(reasons),
    }


def scan_ass_segment_candidate(source_token: str, candidate: str, target_lang: str) -> dict:
    cleaned = strip_label_prefix(candidate or "")
    reasons = []
    if not cleaned.strip():
        reasons.append("empty_output")
    if not ass_placeholders_match(source_token, cleaned):
        reasons.append("placeholder_mismatch")
    if has_ass_markup(cleaned):
        reasons.append("unexpected_ass_markup")
    if has_placeholder_artifacts(cleaned):
        reasons.append("placeholder_artifacts")
    if has_target_language_leak(source_token, cleaned, target_lang):
        reasons.append("language_leak")
    if is_unchanged_english(source_token, cleaned, target_lang):
        reasons.append("unchanged")
    if is_very_suspicious_translation(source_token, cleaned):
        reasons.append("very_suspicious")

    ok = len(reasons) == 0
    needs_llm_repair = any(
        reason in {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged", "language_leak", "very_suspicious"}
        for reason in reasons
    )
    return {
        "ok": ok,
        "candidate": cleaned,
        "reasons": reasons,
        "needs_llm_repair": needs_llm_repair,
        "score": repair_issue_score(reasons),
    }


def self_test_ass_repair_snippet() -> None:
    source_protected = "__TAG_0__ I couldn't make it\\N..."
    bad_candidate = "ASS: I pude atender a\\N..."
    cleaned_bad = strip_label_prefix(bad_candidate)
    assert not has_label_prefix_artifact(cleaned_bad)
    assert has_target_language_leak(source_protected, cleaned_bad, "Spanish")

    fixed_candidate = "__TAG_0__ No pude llegar\\N..."
    cleaned_fixed = strip_label_prefix(fixed_candidate)
    assert not has_label_prefix_artifact(cleaned_fixed)
    assert " i " not in f" {ass_plain_text(cleaned_fixed).lower()} "
    assert not has_target_language_leak(source_protected, cleaned_fixed, "Spanish")

    recover_state = {
        "text_field": "I was constructed with an\\N\"idealized human form\" body concept,",
        "protected": "I was constructed with an__TAG_0__\"idealized human form\" body concept,",
        "replacements": [("__TAG_0__", "\\N")],
        "tokens": ["I was constructed with an", "\\N", "\"idealized human form\" body concept,"],
    }
    recovered = recover_ass_candidate_placeholders(
        recover_state,
        "Yo fui construido con una forma corporal de \"ser humano idealizada\",",
    )
    assert recovered is not None
    recovered_status = scan_ass_line_candidate(recover_state, recovered, "Spanish")
    assert "placeholder_mismatch" not in recovered_status["reasons"]

    # Quick scan test: only the problematic "ASS: I ..." line should require LLM repair.
    texts = [
        "Hello there.",
        "{\\i1}I couldn't make it\\N...",
        "See you soon.",
    ]
    candidates = [
        "Hola por ahi.",
        "ASS: I pude atender a\\N...",
        "Nos vemos pronto.",
    ]
    flagged = []
    for idx, (text_field, cand) in enumerate(zip(texts, candidates)):
        protected, replacements = ass_protect_tags(text_field)
        state = {
            "text_field": text_field,
            "protected": protected,
            "replacements": replacements,
        }
        status = scan_ass_line_candidate(state, cand, "Spanish")
        if status["needs_llm_repair"]:
            flagged.append(idx)
    assert flagged == [1]


def self_test_hybrid_pipeline() -> None:
    assert strip_label_prefix("ASS: Hola") == "Hola"
    src = "{\\i1}do{\\i0}\\Nnow"
    protected, replacements = ass_protect_tags(src)
    candidate = "__TAG_0__hacer__TAG_1____TAG_2__ahora"
    assert ass_placeholders_match(protected, candidate)
    restored = ass_restore_tags(candidate, replacements)
    assert restored.startswith("{\\i1}")
    assert "\\N" in restored
    assert ass_structure_preserved(src, restored)
    assert "ASS:" not in strip_label_prefix("ASS: traduccion correcta")
    srt_source = {"original_text_field": "Hello\nthere", "expected_line_count": 2}
    srt_status = scan_srt_candidate(srt_source, "Hola\nalli", "Spanish")
    assert srt_status["ok"]
    miss_src = "Miss Ito,"
    miss_protected, miss_repl = ass_protect_tags(miss_src)
    miss_state = {
        "text_field": miss_src,
        "protected": miss_protected,
        "replacements": miss_repl,
    }
    miss_status = scan_ass_line_candidate(miss_state, "Miss Ito,", "Spanish")
    assert "unchanged" in miss_status["reasons"]
    assert miss_status["needs_llm_repair"]
    credits = "{\\an8}Brought to you by [ToonsHub]"
    assert apply_phrase_overrides(credits) == "{\\an8}Tra\u00eddo por [el_inmortus]"


def parse_ass_ir(text: str, limit: int | None) -> Tuple[List[str], List[dict]]:
    lines = text.splitlines()
    in_events = False
    format_fields = None
    text_idx = None
    items: List[dict] = []
    translated_count = 0

    for line_idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[Events]"):
            in_events = True
            continue
        if in_events and stripped.startswith("Format:"):
            fmt = stripped[len("Format:") :].strip()
            format_fields = [f.strip() for f in fmt.split(",")]
            if "Text" in format_fields:
                text_idx = format_fields.index("Text")
            continue
        if not in_events or format_fields is None or text_idx is None:
            continue

        m = re.match(r"^(Dialogue|Comment):(\s*)(.*)$", line)
        if not m:
            continue
        kind = m.group(1)
        spaces = m.group(2)
        rest = m.group(3)
        fields = rest.split(",", maxsplit=len(format_fields) - 1)
        if len(fields) != len(format_fields):
            continue
        if kind.lower() != "dialogue":
            continue
        if limit is not None and translated_count >= limit:
            continue

        text_field = fields[text_idx]
        if not should_translate(text_field):
            continue
        fixed_final_text = None
        overridden_text = apply_phrase_overrides(text_field)
        if overridden_text != text_field:
            fixed_final_text = overridden_text
        protected, replacements = ass_protect_tags(text_field)
        tokens = split_ass_text(text_field)
        item = {
            "id": ("ass_dialogue", line_idx),
            "kind": "ass_dialogue",
            "line_idx": line_idx,
            "dialogue_kind": kind,
            "spaces": spaces,
            "fields": fields,
            "text_idx": text_idx,
            "original_text_field": text_field,
            "tokens": tokens,
            "protected_text": protected,
            "text_field": text_field,
            "protected": protected,
            "replacements": replacements,
            "in_tokens": estimate_tokens_from_text(protected),
            "fixed_final_text": fixed_final_text,
        }
        items.append(item)
        translated_count += 1
    return lines, items


def parse_srt_ir(text: str, limit: int | None) -> Tuple[List[List[str]], List[dict]]:
    lines = text.splitlines()
    blocks: List[List[str]] = []
    cur: List[str] = []
    for line in lines:
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
            continue
        cur.append(line)
    if cur:
        blocks.append(cur)

    items: List[dict] = []
    translated_count = 0
    for block_idx, block in enumerate(blocks):
        if len(block) < 2:
            continue
        if limit is not None and translated_count >= limit:
            continue
        text_lines = block[2:] if len(block) > 2 else [""]
        joined = "\n".join(text_lines)
        fixed_final_text = None
        overridden_text = apply_phrase_overrides(joined)
        if overridden_text != joined:
            fixed_final_text = overridden_text
        item = {
            "id": ("srt_block", block_idx),
            "kind": "srt_block",
            "block_idx": block_idx,
            "block": block,
            "original_text_field": joined,
            "raw": joined,
            "protected_text": joined,
            "expected_line_count": len(text_lines),
            "in_tokens": estimate_tokens_from_text(joined),
            "fixed_final_text": fixed_final_text,
        }
        items.append(item)
        translated_count += 1
    return blocks, items


def enforce_line_count(text: str, expected_count: int) -> str:
    if expected_count <= 1:
        return text.replace("\n", " ").strip()
    lines = text.split("\n")
    if len(lines) == expected_count:
        return text
    merged = " ".join(part.strip() for part in lines if part.strip()).strip()
    if not merged:
        return "\n".join([""] * expected_count)
    words = merged.split()
    if len(words) < expected_count:
        words += [""] * (expected_count - len(words))
    chunk = int(math.ceil(len(words) / float(expected_count)))
    out_lines = []
    cursor = 0
    for _ in range(expected_count):
        out_lines.append(" ".join(words[cursor : cursor + chunk]).strip())
        cursor += chunk
    return "\n".join(out_lines)


def scan_srt_candidate(item: dict, candidate: str, target_lang: str) -> dict:
    cleaned = strip_label_prefix(candidate or "")
    source_text = item["original_text_field"]
    reasons = []
    if not cleaned.strip():
        reasons.append("empty_output")
    if has_label_prefix_artifact(cleaned):
        reasons.append("label_prefix")
    expected_line_count = item.get("expected_line_count", 1)
    if cleaned.count("\n") + 1 != expected_line_count:
        reasons.append("line_break_mismatch")
    if is_unchanged_english(source_text, cleaned, target_lang):
        reasons.append("unchanged")
    if has_target_language_leak(source_text, cleaned, target_lang):
        reasons.append("language_leak")
    if is_very_suspicious_translation(source_text, cleaned):
        reasons.append("very_suspicious")

    needs_repair = any(
        reason in {"empty_output", "label_prefix", "line_break_mismatch", "unchanged", "language_leak", "very_suspicious"}
        for reason in reasons
    )
    return {
        "ok": not reasons,
        "candidate": cleaned,
        "reasons": reasons,
        "needs_llm_repair": needs_repair,
        "score": repair_issue_score(reasons),
    }


def extract_ass_slots(tokens: List[str]) -> Tuple[List[int], List[str]]:
    slot_positions = []
    slots = []
    for idx, token in enumerate(tokens):
        if token.startswith("{") or token in ("\\N", "\\n", "\\h"):
            continue
        slot_positions.append(idx)
        slots.append(token)
    return slot_positions, slots


def reinsert_ass_slots(tokens: List[str], slot_positions: List[int], translated_slots: List[str]) -> str:
    rebuilt = list(tokens)
    for pos, translated in zip(slot_positions, translated_slots):
        rebuilt[pos] = translated
    return "".join(rebuilt)


def translate_ass_slots_single(
    client: OllamaClient,
    item: dict,
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
) -> str | None:
    slot_positions, slots = extract_ass_slots(item["tokens"])
    if not slots:
        return item["original_text_field"]
    summary_short, tone_short = trim_translation_context(summary, tone_guide)
    full_plain = ass_plain_text(item["original_text_field"]).replace("\n", " ").strip()
    user_msg = (
        f"Translate slot texts to {target_lang}.\n"
        "Rules:\n"
        "1) Return ONLY a JSON array of strings with the same length/order as `slots`.\n"
        "2) Do NOT add labels like ASS:, SRT:, Dialogue:.\n"
        "3) Do not translate content inside square brackets [like this]; preserve it exactly.\n"
        "4) Do NOT add braces/tags/control codes.\n"
        "5) Output must be compact/minified JSON (single line, no extra whitespace/newlines).\n"
    )
    if summary_short:
        user_msg += f"Summary hint: {summary_short}\n"
    if tone_short:
        user_msg += f"Tone hint: {tone_short}\n"
    user_msg += f"Full line plain text: {full_plain}\nInput slots JSON: {dump_json(slots)}"
    try:
        schema = build_fixed_array_schema(len(slots))
        options_batch = build_options_for_batch(options, slots, force_temperature_zero=True)
        with RUNTIME_METRICS.timed("retry.chat.ass_slot_single"):
            content = client.chat(
                [
                    {"role": "system", "content": "You translate subtitle slots. Output only compact/minified JSON."},
                    {"role": "user", "content": user_msg},
                ],
                options=options_batch,
                format=schema,
            )
    except OllamaTimeoutError:
        return None
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            return None
        return None
    out = parse_array_response(content, len(slots))
    if out is None:
        return None
    rebuilt = reinsert_ass_slots(item["tokens"], slot_positions, [strip_label_prefix(s) for s in out])
    if not ass_structure_preserved(item["original_text_field"], rebuilt):
        return None
    return rebuilt


def translate_ass_slots_batch_with_split(
    client: OllamaClient,
    items: List[dict],
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
) -> List[str | None]:
    if not items:
        return []
    payload = []
    payload_meta = []
    for item in items:
        slot_positions, slots = extract_ass_slots(item["tokens"])
        payload_meta.append((slot_positions, slots))
        payload.append(
            {
                "full": ass_plain_text(item["original_text_field"]).replace("\n", " ").strip(),
                "slots": slots,
            }
        )
    summary_short, tone_short = trim_translation_context(summary, tone_guide)
    user_msg = (
        f"Translate each `slots` array to {target_lang} using `full` only as context.\n"
        "Rules:\n"
        "1) Output ONLY a JSON array of arrays of strings.\n"
        "2) Outer array length must equal input length; each inner array must match its `slots` length.\n"
        "3) Do NOT add labels like ASS:, SRT:, Dialogue:.\n"
        "4) Do not translate content inside square brackets [like this]; preserve it exactly.\n"
        "5) Do NOT add braces/tags/control codes.\n"
        "6) Output must be compact/minified JSON (single line, no extra whitespace/newlines).\n"
    )
    if summary_short:
        user_msg += f"Summary hint: {summary_short}\n"
    if tone_short:
        user_msg += f"Tone hint: {tone_short}\n"
    user_msg += f"Input JSON: {dump_json(payload)}"

    try:
        schema = build_fixed_array_of_arrays_schema(len(items))
        options_batch = build_options_for_batch(
            options,
            [dump_json(entry) for entry in payload],
            force_temperature_zero=True,
        )
        with RUNTIME_METRICS.timed("retry.chat.ass_slot_batch"):
            content = client.chat(
                [
                    {"role": "system", "content": "You translate subtitle slots. Output only compact/minified JSON."},
                    {"role": "user", "content": user_msg},
                ],
                options=options_batch,
                format=schema,
            )
    except OllamaTimeoutError:
        content = None
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            content = None
        else:
            content = None

    if content is not None:
        parsed = parse_array_of_arrays_response(content, len(items))
    else:
        parsed = None

    if parsed is not None and len(parsed) == len(items):
        out_lines: List[str | None] = []
        for item, row, (slot_positions, slots) in zip(items, parsed, payload_meta):
            if len(row) != len(slots):
                out_lines.append(None)
                continue
            rebuilt = reinsert_ass_slots(
                item["tokens"],
                slot_positions,
                [strip_label_prefix(text) for text in row],
            )
            if not ass_structure_preserved(item["original_text_field"], rebuilt):
                out_lines.append(None)
                continue
            out_lines.append(rebuilt)
        return out_lines

    if len(items) == 1:
        return [translate_ass_slots_single(client, items[0], summary, tone_guide, target_lang, options)]
    mid = max(1, len(items) // 2)
    return (
        translate_ass_slots_batch_with_split(client, items[:mid], summary, tone_guide, target_lang, options)
        + translate_ass_slots_batch_with_split(client, items[mid:], summary, tone_guide, target_lang, options)
    )


def run_ass_surgical_fallback(
    client: OllamaClient,
    failed_items: List[dict],
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
    console,
) -> None:
    if not failed_items:
        return
    RUNTIME_METRICS.bump("retry.ass_surgical.attempts")
    RUNTIME_METRICS.bump("retry.ass_surgical.items", len(failed_items))
    total = len(failed_items)
    if total <= 6:
        for item in failed_items:
            repaired = translate_ass_slots_single(client, item, summary, tone_guide, target_lang, options)
            if repaired is not None:
                item["forced_restored"] = repaired
        return
    if total > 20:
        cprint(console, f"Large ASS repair set ({total}); applying chunked surgical fallback.", "yellow")
    for chunk in batched(failed_items, 8):
        repaired_lines = translate_ass_slots_batch_with_split(
            client,
            chunk,
            summary,
            tone_guide,
            target_lang,
            options,
        )
        for item, repaired in zip(chunk, repaired_lines):
            if repaired is not None:
                item["forced_restored"] = repaired


def fallback_srt_linewise(
    client: OllamaClient,
    item: dict,
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
) -> str | None:
    RUNTIME_METRICS.bump("retry.srt_linewise.attempts")
    source_lines = item["original_text_field"].split("\n")
    RUNTIME_METRICS.bump("retry.srt_linewise.lines", len(source_lines))
    out_lines = translate_json_batch_with_split(
        client,
        source_lines,
        summary,
        tone_guide,
        target_lang,
        options,
        mode="srt_block",
        rolling_context="",
        timing_label="retry.chat.srt_linewise",
        format_mode="schema",
    )
    if len(out_lines) != len(source_lines):
        return None
    cleaned_lines = []
    for source, translated in zip(source_lines, out_lines):
        if translated is None:
            return None
        cleaned = strip_label_prefix(translated)
        if has_target_language_leak(source, cleaned, target_lang):
            return None
        cleaned_lines.append(cleaned)
    return "\n".join(cleaned_lines)


def retry_single_item_translation(
    client: OllamaClient,
    source_text: str,
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
    mode: str,
) -> str | None:
    RUNTIME_METRICS.bump("retry.single_item.attempts")
    out = translate_json_batch_with_split(
        client,
        [source_text],
        summary,
        tone_guide,
        target_lang,
        options,
        mode=mode,
        rolling_context="",
        force_temperature_zero=True,
        timing_label="retry.chat.single_item",
        format_mode="schema",
    )
    if not out or out[0] is None:
        return None
    return strip_label_prefix(out[0])


def retry_failed_items_batch(
    client: OllamaClient,
    failed_items: List[dict],
    summary: str,
    tone_guide: str,
    target_lang: str,
    options,
    mode: str,
    format_mode: str | None = "schema",
) -> None:
    """Reintenta en batch (temp=0 + schema) para evitar N llamadas 1x1 cuando el batch salió muy mal."""
    if not failed_items:
        return
    RUNTIME_METRICS.bump("retry.batch_temp0.attempts")
    RUNTIME_METRICS.bump("retry.batch_temp0.items", len(failed_items))
    sources = [it["protected_text"] for it in failed_items]
    out = translate_json_batch_with_split(
        client,
        sources,
        summary,
        tone_guide,
        target_lang,
        options,
        mode=mode,
        rolling_context="",
        force_temperature_zero=True,
        timing_label="retry.chat.batch_temp0",
        format_mode=format_mode,
    )
    for it, cand in zip(failed_items, out):
        if cand is None:
            continue
        if mode == "ass_protected":
            it["text_field"] = it["original_text_field"]
            it["protected"] = it["protected_text"]
            status = scan_ass_line_candidate(it, cand, target_lang)
            it["candidate"] = status["candidate"]
            it["status"] = status
        else:
            status = scan_srt_candidate(it, cand, target_lang)
            it["candidate"] = status["candidate"]
            it["status"] = status


def translate_srt(
    client: OllamaClient,
    text: str,
    summary: str,
    tone_guide: str,
    target_lang: str,
    batch_size: int,
    options,
    limit: int | None,
    console,
    one_shot: bool = False,
    rolling_context: int = 0,
) -> Tuple[str, int]:
    blocks, items = parse_srt_ir(text, limit)
    if not items:
        return text, 0

    translatable_items = [item for item in items if not item.get("fixed_final_text")]
    for item in items:
        if item.get("fixed_final_text"):
            item["final_text"] = item["fixed_final_text"]

    ass_batch_cap = effective_batch_cap(batch_size, one_shot)
    batches = build_adaptive_batches(translatable_items, ass_batch_cap, options, one_shot, console)
    history: List[str] = []
    with progress_bar(console) as progress:
        task_id = progress.add_task("Translation", total=len(translatable_items))
        for batch in batches:
            RUNTIME_METRICS.bump("translate.top_level_batches", 1)
            RUNTIME_METRICS.bump("translate.top_level_batch_items", len(batch))
            sources = [item["protected_text"] for item in batch]
            context_hint = rolling_context_snippet(history, rolling_context)
            out = translate_json_batch_with_split(
                client,
                sources,
                summary,
                tone_guide,
                target_lang,
                options,
                mode="srt_block",
                rolling_context=context_hint,
            )
            for item, candidate in zip(batch, out):
                status = scan_srt_candidate(item, candidate or "", target_lang)
                item["candidate"] = status["candidate"]
                item["status"] = status

            failed_items = [item for item in batch if item["status"]["needs_llm_repair"]]
            repair_limit = max(8, int(math.ceil(0.10 * max(1, len(batch)))))
            if failed_items and len(failed_items) > repair_limit:
                cprint(
                    console,
                    f"High fail-rate: {len(failed_items)}/{len(batch)} SRT flagged. Retrying batch @temp=0...",
                    "yellow",
                )
                retry_failed_items_batch(
                    client,
                    failed_items,
                    summary,
                    tone_guide,
                    target_lang,
                    options,
                    mode="srt_block",
                )
                failed_items = [item for item in batch if item["status"]["needs_llm_repair"]]
            if failed_items:
                for item in failed_items:
                    repaired = fallback_srt_linewise(
                        client,
                        item,
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                    )
                    if repaired is not None:
                        item["forced_text"] = repaired

            for item in batch:
                if "forced_text" in item:
                    final_text = item["forced_text"]
                elif item["status"]["ok"]:
                    final_text = enforce_line_count(
                        item["status"]["candidate"],
                        item["expected_line_count"],
                    )
                else:
                    retried = retry_single_item_translation(
                        client,
                        item["original_text_field"],
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                        mode="srt_block",
                    )
                    if retried is not None:
                        retry_status = scan_srt_candidate(item, retried, target_lang)
                        if retry_status["ok"]:
                            final_text = enforce_line_count(
                                retry_status["candidate"],
                                item["expected_line_count"],
                            )
                        else:
                            final_text = item["original_text_field"]
                    else:
                        final_text = item["original_text_field"]
                item["final_text"] = final_text
                history.append(final_text.replace("\n", " ").strip())
            progress.advance(task_id, len(batch))

    for item in items:
        block = blocks[item["block_idx"]]
        final_text = apply_phrase_overrides(item["final_text"])
        new_lines = final_text.split("\n")
        blocks[item["block_idx"]] = [block[0], block[1], *new_lines]

    out_lines = []
    for i, block in enumerate(blocks):
        out_lines.extend(block)
        if i != len(blocks) - 1:
            out_lines.append("")
    return "\n".join(out_lines), len(items)


def translate_ass(
    client: OllamaClient,
    text: str,
    summary: str,
    tone_guide: str,
    target_lang: str,
    batch_size: int,
    options,
    limit: int | None,
    console,
    ass_mode: str = "line",
    one_shot: bool = False,
    rolling_context: int = 0,
) -> Tuple[str, int]:
    if ass_mode == "segment":
        return translate_ass_segment(
            client,
            text,
            summary,
            tone_guide,
            target_lang,
            batch_size,
            options,
            limit,
            console,
        )
    lines, items = parse_ass_ir(text, limit)
    if not items:
        return text, 0

    translatable_items = [item for item in items if not item.get("fixed_final_text")]
    for item in items:
        if item.get("fixed_final_text"):
            item["final_text"] = item["fixed_final_text"]

    ass_batch_cap = effective_batch_cap(batch_size, one_shot)
    batches = build_adaptive_batches(translatable_items, ass_batch_cap, options, one_shot, console)
    severe_reasons = {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged"}
    history: List[str] = []

    with progress_bar(console) as progress:
        task_id = progress.add_task("Translation", total=len(translatable_items))
        for batch in batches:
            RUNTIME_METRICS.bump("translate.top_level_batches", 1)
            RUNTIME_METRICS.bump("translate.top_level_batch_items", len(batch))
            sources = [item["protected_text"] for item in batch]
            context_hint = rolling_context_snippet(history, rolling_context)
            out = translate_json_batch_with_split(
                client,
                sources,
                summary,
                tone_guide,
                target_lang,
                options,
                mode="ass_protected",
                rolling_context=context_hint,
            )
            for item, candidate in zip(batch, out):
                candidate_text = candidate if candidate is not None else item["protected_text"]
                item["text_field"] = item["original_text_field"]
                item["protected"] = item["protected_text"]
                status = scan_ass_line_candidate(item, candidate_text, target_lang)
                item["candidate"] = status["candidate"]
                item["status"] = status

            failed_items = [item for item in batch if item["status"]["needs_llm_repair"]]
            repair_limit = max(8, int(math.ceil(0.10 * max(1, len(batch)))))
            if failed_items:
                if len(failed_items) > repair_limit:
                    cprint(
                        console,
                        f"High fail-rate: {len(failed_items)}/{len(batch)} ASS flagged. Retrying batch @temp=0...",
                        "yellow",
                    )
                retry_failed_items_batch(
                    client,
                    failed_items,
                    summary,
                    tone_guide,
                    target_lang,
                    options,
                    mode="ass_protected",
                    format_mode="schema",
                )
                failed_items = [item for item in batch if item["status"]["needs_llm_repair"]]
            if failed_items:
                run_ass_surgical_fallback(client, failed_items, summary, tone_guide, target_lang, options, console)

            for item in batch:
                status = item["status"]
                if "forced_restored" in item:
                    forced = item["forced_restored"]
                    if ass_structure_preserved(item["original_text_field"], forced):
                        final_text = forced
                    else:
                        final_text = item["original_text_field"]
                elif status["ok"] or not any(reason in severe_reasons for reason in status["reasons"]):
                    final_text = status["restored"]
                else:
                    retried = retry_single_item_translation(
                        client,
                        item["protected_text"],
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                        mode="ass_protected",
                    )
                    if retried is not None:
                        retry_status = scan_ass_line_candidate(item, retried, target_lang)
                        if retry_status["ok"]:
                            final_text = retry_status["restored"]
                        else:
                            final_text = item["original_text_field"]
                    else:
                        final_text = item["original_text_field"]
                item["final_text"] = final_text
                history.append(ass_plain_text(final_text).replace("\n", " ").strip())
            progress.advance(task_id, len(batch))

    for item in items:
        fields = item["fields"]
        fields[item["text_idx"]] = apply_phrase_overrides(item["final_text"])
        lines[item["line_idx"]] = f"{item['dialogue_kind']}:{item['spaces']}{','.join(fields)}"

    return "\n".join(lines), len(items)


def translate_ass_segment(
    client: OllamaClient,
    text: str,
    summary: str,
    tone_guide: str,
    target_lang: str,
    batch_size: int,
    options,
    limit: int | None,
    console,
) -> Tuple[str, int]:
    lines = text.splitlines()
    in_events = False
    format_fields = None
    text_idx = None

    trans_tasks = []
    line_state = {}
    translated_count = 0

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("[Events]"):
            in_events = True
            continue
        if in_events and stripped.startswith("Format:"):
            fmt = stripped[len("Format:") :].strip()
            format_fields = [f.strip() for f in fmt.split(",")]
            if "Text" in format_fields:
                text_idx = format_fields.index("Text")
            continue

        if not in_events or format_fields is None or text_idx is None:
            continue

        m = re.match(r"^(Dialogue|Comment):(\s*)(.*)$", line)
        if not m:
            continue
        kind = m.group(1)
        spaces = m.group(2)
        rest = m.group(3)
        fields = rest.split(",", maxsplit=len(format_fields) - 1)
        if len(fields) != len(format_fields):
            continue
        if kind.lower() != "dialogue":
            continue
        if limit is not None and translated_count >= limit:
            continue

        translated_count += 1
        text_field = fields[text_idx]
        tokens = split_ass_text(text_field)
        line_state[i] = {
            "kind": kind,
            "spaces": spaces,
            "fields": fields,
            "tokens": tokens,
            "text_idx": text_idx,
        }
        for t_idx, token in enumerate(tokens):
            if token.startswith("{") or token in ("\\N", "\\n", "\\h"):
                continue
            if should_translate(token):
                trans_tasks.append((i, t_idx, token))

    translations = []
    if trans_tasks:
        segments = [t[2] for t in trans_tasks]
        with progress_bar(console) as progress:
            task_id = progress.add_task("Translation", total=len(segments))
            for batch in batched(segments, batch_size):
                out = translate_batch(client, batch, summary, tone_guide, target_lang, options)
                if out is None or len(out) != len(batch):
                    out = []
                    for item in batch:
                        out.append(translate_one(client, item, summary, tone_guide, target_lang, options))
                translations.extend(out)
                progress.advance(task_id, len(batch))
        cprint(console, "Applying ASS safety checks...", "cyan")

    for (line_idx, token_idx, source_token), translated in zip(trans_tasks, translations):
        state = line_state.get(line_idx)
        if not state:
            continue
        state.setdefault("task_candidates", {})
        state["task_candidates"][token_idx] = strip_label_prefix(translated)

    for line_idx, token_idx, source_token in trans_tasks:
        state = line_state.get(line_idx)
        if not state:
            continue
        state.setdefault("task_candidates", {})
        if token_idx not in state["task_candidates"]:
            state["task_candidates"][token_idx] = source_token
        state.setdefault("task_status", {})
        candidate = state["task_candidates"][token_idx]
        state["task_status"][token_idx] = scan_ass_segment_candidate(source_token, candidate, target_lang)

    failed_tasks: List[Tuple[int, int, str]] = []
    for line_idx, token_idx, source_token in trans_tasks:
        state = line_state.get(line_idx)
        if not state:
            continue
        status = state.get("task_status", {}).get(token_idx, {})
        if status.get("needs_llm_repair"):
            failed_tasks.append((line_idx, token_idx, source_token))

    repair_limit = max(8, int(math.ceil(0.15 * max(1, len(trans_tasks)))))
    allow_llm_repair = len(failed_tasks) <= repair_limit
    if failed_tasks and not allow_llm_repair:
        cprint(console, "Too many lines flagged; skipping LLM repair to keep performance", "yellow")

    if failed_tasks and allow_llm_repair:
        repair_items = []
        for line_idx, token_idx, source_token in failed_tasks:
            state = line_state.get(line_idx)
            candidate = ""
            if state:
                candidate = state.get("task_candidates", {}).get(token_idx, "")
            repair_items.append(
                {
                    "src": source_token,
                    "bad": candidate,
                }
            )
        repaired = repair_batch(client, repair_items, summary, tone_guide, target_lang, options)
        for (line_idx, token_idx, _), candidate in zip(failed_tasks, repaired):
            if candidate is None:
                continue
            state = line_state.get(line_idx)
            if state:
                state.setdefault("task_candidates", {})
                state["task_candidates"][token_idx] = strip_label_prefix(candidate)
                state.setdefault("task_status", {})
                state["task_status"][token_idx] = scan_ass_segment_candidate(source_token, state["task_candidates"][token_idx], target_lang)

    unresolved_tasks = []
    for line_idx, token_idx, source_token in failed_tasks:
        state = line_state.get(line_idx)
        if not state:
            continue
        status = state.get("task_status", {}).get(token_idx, {})
        if status.get("needs_llm_repair"):
            unresolved_tasks.append((line_idx, token_idx, source_token))

    unresolved_tasks = sorted(
        unresolved_tasks,
        key=lambda item: line_state[item[0]]["task_status"][item[1]]["score"],
        reverse=True,
    )[:MAX_REPAIR_FALLBACK_LINES]

    for line_idx, token_idx, source_token in unresolved_tasks:
        state = line_state.get(line_idx)
        if not state:
            continue
        candidate = state["task_candidates"][token_idx]
        status = state["task_status"][token_idx]
        if status.get("needs_llm_repair"):
            retry = guarded_translate_one(
                client,
                source_token,
                summary,
                tone_guide,
                target_lang,
                options,
                source_for_checks=source_token,
                validator=lambda cand, src=source_token: scan_ass_segment_candidate(src, cand, target_lang)["ok"],
                prefer_no_context=True,
                chat_timing_label="retry.chat.segment_single",
            )
            if retry is not None:
                retry = strip_label_prefix(retry)
                status = scan_ass_segment_candidate(source_token, retry, target_lang)
                if status["ok"]:
                    candidate = retry
        state["task_candidates"][token_idx] = candidate
        state["task_status"][token_idx] = status

    severe_reasons = {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged"}
    for line_idx, token_idx, source_token in trans_tasks:
        state = line_state.get(line_idx)
        if not state:
            continue
        candidate = state["task_candidates"][token_idx]
        status = state["task_status"][token_idx]
        if status["ok"] or not any(reason in severe_reasons for reason in status["reasons"]):
            state["tokens"][token_idx] = candidate
        else:
            state["tokens"][token_idx] = source_token

    for line_idx, state in line_state.items():
        new_text = apply_phrase_overrides("".join(state["tokens"]))
        fields = state["fields"]
        fields[state["text_idx"]] = new_text
        lines[line_idx] = f"{state['kind']}:{state['spaces']}{','.join(fields)}"

    return "\n".join(lines), translated_count


def collect_plain_lines_ass(text: str) -> List[str]:
    lines = []
    in_events = False
    format_fields = None
    text_idx = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("[Events]"):
            in_events = True
            continue
        if in_events and stripped.startswith("Format:"):
            fmt = stripped[len("Format:") :].strip()
            format_fields = [f.strip() for f in fmt.split(",")]
            if "Text" in format_fields:
                text_idx = format_fields.index("Text")
            continue
        if not in_events or format_fields is None or text_idx is None:
            continue
        if not stripped.startswith("Dialogue:"):
            continue
        rest = stripped[len("Dialogue:") :].lstrip()
        fields = rest.split(",", maxsplit=len(format_fields) - 1)
        if len(fields) != len(format_fields):
            continue
        text_field = fields[text_idx]
        lines.append(ass_plain_text(text_field))
    return lines


def collect_plain_lines_srt(text: str) -> List[str]:
    lines = []
    blocks = text.split("\n\n")
    for block in blocks:
        parts = block.splitlines()
        if len(parts) >= 3:
            lines.extend(parts[2:])
    return lines


def detect_language(text: str) -> Tuple[str, float]:
    lowered = text.lower()
    tokens = re.findall(r"[a-záéíóúñü]+", lowered)
    en_count = sum(1 for t in tokens if t in EN_STOPWORDS)
    es_count = sum(1 for t in tokens if t in ES_STOPWORDS)
    accent_bonus = sum(1 for ch in lowered if ch in "áéíóúñü")
    es_count += accent_bonus
    total = en_count + es_count
    if total == 0:
        return "Unknown", 0.0
    if en_count > es_count:
        lang = "English"
    elif es_count > en_count:
        lang = "Spanish"
    else:
        lang = "Unknown"
    confidence = abs(en_count - es_count) / total
    return lang, confidence


def bulk_dir() -> Path:
    return Path.cwd() / BULK_DIR_NAME


def resolve_input_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return bulk_dir() / path


def resolve_output_path(path_like: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute() or path.parent != Path("."):
        return path
    return bulk_dir() / path


def list_subtitle_files(exts: List[str]) -> List[Path]:
    base_dir = bulk_dir()
    if not base_dir.exists():
        return []
    files = []
    for ext in exts:
        files.extend(base_dir.glob(f"*{ext}"))
        files.extend(base_dir.glob(f"*{ext.upper()}"))
    return sorted(set(files))


def get_installed_models_list() -> List[str]:
    try:
        proc = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        if proc.returncode != 0:
            return []
        lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            return []
        names = []
        for line in lines[1:]:
            parts = line.split()
            if parts:
                names.append(parts[0])
        return names
    except Exception:
        return []


def choose_model(console, default_model: str) -> str:
    installed = get_installed_models_list()
    ordered = installed[:]
    if "gemma3:4b" in ordered:
        ordered.remove("gemma3:4b")
        ordered.insert(0, "gemma3:4b")

    if not ordered:
        cprint(console, "No installed models found via `ollama list`.", "yellow")
        return default_model or input("Enter model name: ").strip()

    cprint(console, "Choose model:", "bold cyan")
    for i, name in enumerate(ordered, 1):
        console.print(f"  {i}) {name}")
    console.print("  0) Enter custom model")

    default_idx = None
    if default_model in ordered:
        default_idx = ordered.index(default_model) + 1

    prompt = "Model"
    while True:
        suffix = f" [default {default_idx}]" if default_idx else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default_idx:
            return ordered[default_idx - 1]
        if raw.isdigit():
            num = int(raw)
            if num == 0:
                custom = input("Enter model name: ").strip()
                if custom:
                    return custom
            elif 1 <= num <= len(ordered):
                return ordered[num - 1]
        elif raw:
            return raw
        cprint(console, "Invalid choice. Try again.", "yellow")


def prompt_choice(
    console,
    prompt: str,
    max_value: int,
    default_value: int | None = None,
) -> int:
    while True:
        suffix = f" [default {default_value}]" if default_value else ""
        choice = input(f"{prompt}{suffix}: ").strip()
        if choice == "" and default_value is not None:
            return default_value
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= max_value:
                return num
        cprint(console, "Invalid choice. Try again.", "yellow")


def target_suffix_for_lang(target_lang: str) -> str:
    lang = target_lang.strip().lower()
    if lang in ("spanish", "es", "es-419", "es_419"):
        return "es-419"
    if lang in ("english", "en", "en-us", "en_us", "en-gb", "en_gb"):
        return "en"
    return "translated"


def is_already_target_file(path: Path, target_lang: str) -> bool:
    stem = path.stem.lower()
    suffix = target_suffix_for_lang(target_lang).lower()
    if suffix == "translated":
        return stem.endswith(".translated")
    return stem.endswith(f"_{suffix}")


def build_output_path(in_path: Path, target_lang: str) -> Path:
    suffix = in_path.suffix
    stem = in_path.stem
    out_dir = bulk_dir()
    target_suffix = target_suffix_for_lang(target_lang)

    normalized = stem.lower()
    if target_suffix == "translated":
        if normalized.endswith(".translated"):
            stem = stem[: -len(".translated")]
        return out_dir / f"{stem}.translated{suffix}"

    if normalized.endswith(f"_{target_suffix.lower()}"):
        stem = stem[: -(len(target_suffix) + 1)]
    return out_dir / f"{stem}_{target_suffix}{suffix}"


def collect_batch_inputs(raw_input: str | None, target_lang: str) -> List[Path]:
    files = list_subtitle_files([".srt", ".ass"])
    if raw_input:
        resolved = resolve_input_path(raw_input)
        if resolved.exists():
            files = [resolved]
        else:
            files = [f for f in files if fnmatch.fnmatch(f.name, raw_input)]

    out = []
    seen = set()
    for file_path in files:
        if is_already_target_file(file_path, target_lang):
            continue
        if file_path not in seen:
            seen.add(file_path)
            out.append(file_path)
    return sorted(out)


def interactive_flow(args, console) -> Tuple[Path, Path, str, int | None, bool, str]:
    files = list_subtitle_files([".srt"])
    if not files:
        files = list_subtitle_files([".ass"])
    in_path = None
    if files:
        cprint(console, "Subtitle files found:", "bold cyan")
        for i, f in enumerate(files, 1):
            console.print(f"  {i}) {f.name}")
        raw = input("Select a file number or type a path: ").strip()
        if raw.isdigit():
            idx = int(raw)
            if 1 <= idx <= len(files):
                in_path = files[idx - 1]
        elif raw:
            in_path = resolve_input_path(raw)
        else:
            if len(files) == 1:
                in_path = files[0]
    if in_path is None:
        raw = input("Enter subtitle file path: ").strip()
        in_path = resolve_input_path(raw)
    if not in_path.exists():
        raise RuntimeError(f"Input not found: {in_path}")

    text, _, _, _ = read_text(in_path)
    ext = in_path.suffix.lower()
    if ext == ".ass":
        sample_lines = collect_plain_lines_ass(text)
    else:
        sample_lines = collect_plain_lines_srt(text)

    cprint(console, "\nSample (first 8 lines):", "bold cyan")
    for line in sample_lines[:8]:
        console.print(f"  {line}")

    sample_blob = "\n".join(sample_lines[:50])
    detected, confidence = detect_language(sample_blob)
    conf_pct = int(confidence * 100)
    cprint(
        console,
        f"\nDetected input language: {detected} ({conf_pct}% confidence)",
        "bold green",
    )

    default_out = 1
    if detected == "Spanish":
        default_out = 2
    cprint(console, "Choose output language:", "bold cyan")
    console.print("  1) Spanish")
    console.print("  2) English")
    out_choice = prompt_choice(console, "Output language", 2, default_out)
    target_lang = "Spanish" if out_choice == 1 else "English"

    cprint(console, "Translate sample or full?", "bold cyan")
    console.print("  1) Sample (10 lines)")
    console.print("  2) Full")
    scope_choice = prompt_choice(console, "Scope", 2, 2)
    limit = 10 if scope_choice == 1 else None
    skip_summary = scope_choice == 1

    model = choose_model(console, args.model)

    out_path = resolve_output_path(args.out_path) if args.out_path else build_output_path(in_path, target_lang)
    cprint(console, f"Output file: {out_path}", "bold cyan")
    if skip_summary:
        cprint(console, "Sample mode: skipping summary for speed.", "yellow")
    return in_path, out_path, target_lang, limit, skip_summary, model


def apply_fast_profile(args, console) -> None:
    if not args.fast:
        return
    cpu_threads = os.cpu_count() or 8
    args.ass_mode = "line"
    if args.batch_size < 12:
        args.batch_size = 12
    if args.summary_chars > 3000:
        args.summary_chars = 3000
    if args.temperature > 0.05:
        args.temperature = 0.0
    if args.num_predict is None:
        args.num_predict = 256
    if args.num_ctx is None:
        args.num_ctx = 4096
    if args.num_threads is None:
        args.num_threads = cpu_threads
    if args.rolling_context > 0:
        args.rolling_context = 0
    if not args.skip_summary:
        args.skip_summary = True
    gpu_display = args.num_gpu if args.num_gpu is not None else "auto"
    cprint(
        console,
        f"Fast profile: batch={args.batch_size}, ctx={args.num_ctx}, predict={args.num_predict}, "
        f"threads={args.num_threads}, gpu={gpu_display}, rolling={args.rolling_context}, summary=off",
        "bold cyan",
    )


def build_ollama_options(args) -> dict:
    options = {"temperature": args.temperature}
    if args.num_predict is not None:
        options["num_predict"] = args.num_predict
    if args.num_ctx is not None:
        options["num_ctx"] = args.num_ctx
    if args.num_threads is not None:
        options["num_thread"] = args.num_threads
    if args.num_gpu is not None:
        options["num_gpu"] = args.num_gpu
    return options


def translate_single_file(client, console, args, in_path: Path, out_path: Path) -> int:
    RUNTIME_METRICS.reset()
    text, line_ending, final_newline, bom = read_text(in_path)
    ext = in_path.suffix.lower()
    options = build_ollama_options(args)
    start_total = time.perf_counter()

    summary = ""
    tone_guide = ""
    if not args.skip_summary:
        cprint(console, "Building summary...", "bold cyan")
        with RUNTIME_METRICS.timed("stage.summary"):
            if ext == ".ass":
                plain_lines = collect_plain_lines_ass(text)
            elif ext == ".srt":
                plain_lines = collect_plain_lines_srt(text)
            else:
                print("Unsupported file type. Use .ass or .srt", file=sys.stderr)
                return 2
            summary = summarize_subs(client, plain_lines, args.summary_chars, options, console)
        cprint(console, "Summary ready.", "bold green")
        with RUNTIME_METRICS.timed("stage.tone_guide"):
            tone_guide = build_tone_guide(client, summary, plain_lines, options, console)
        if tone_guide:
            cprint(console, "Tone guide ready.", "bold green")

    cprint(console, "Translating...", "bold cyan")
    with RUNTIME_METRICS.timed("stage.translate"):
        if ext == ".ass":
            out_text, translated_count = translate_ass(
                client,
                text,
                summary,
                tone_guide,
                args.target,
                args.batch_size,
                options,
                args.limit,
                console,
                args.ass_mode,
                args.one_shot,
                args.rolling_context,
            )
        elif ext == ".srt":
            out_text, translated_count = translate_srt(
                client,
                text,
                summary,
                tone_guide,
                args.target,
                args.batch_size,
                options,
                args.limit,
                console,
                args.one_shot,
                args.rolling_context,
            )
        else:
            print("Unsupported file type. Use .ass or .srt", file=sys.stderr)
            return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(out_path, out_text.splitlines(), line_ending, final_newline, bom)
    translate_elapsed = RUNTIME_METRICS.seconds.get("stage.translate", 0.0)
    total_elapsed = time.perf_counter() - start_total
    cprint(console, f"Translated blocks: {translated_count}", "bold green")
    cprint(console, f"Output written to: {out_path}", "bold green")
    cprint(console, f"Elapsed (translate): {translate_elapsed:.1f}s", "bold green")
    cprint(console, f"Elapsed (total): {total_elapsed:.1f}s", "bold green")
    print_runtime_breakdown(console, total_elapsed)
    return 0


def main() -> int:
    console = get_console()
    RUNTIME_METRICS.reset()
    parser = argparse.ArgumentParser(description="Translate .ASS/.SRT subtitles using local Ollama.")
    parser.add_argument("--in", dest="in_path", help="Input subtitle file (or glob in --batch)")
    parser.add_argument("--out", dest="out_path", help="Output subtitle file")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--target", default="Spanish", help="Target language")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size upper bound for translation")
    parser.add_argument("--summary-chars", type=int, default=6000, help="Max chars per summary chunk")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds")
    parser.add_argument("--keep-alive", default="10m", help="Ollama keep_alive value (e.g. 10m, 0)")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--num-predict", type=int, help="Limit tokens generated per response")
    parser.add_argument("--num-ctx", type=int, help="Context window size")
    parser.add_argument("--num-threads", type=int, help="Threads for model execution")
    parser.add_argument("--num-gpu", type=int, help="GPU layers for model execution")
    parser.add_argument("--limit", type=int, help="Translate only the first N dialogue blocks")
    parser.add_argument("--skip-summary", action="store_true", help="Skip the summary step")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--batch", action="store_true", help="Translate all subtitle files in SUBS_BULK")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite outputs in --batch mode")
    parser.add_argument("--ass-mode", choices=["line", "segment"], default="line", help="ASS translation mode")
    parser.add_argument("--fast", action="store_true", help="Apply fast profile defaults")
    parser.add_argument("--one-shot", action="store_true", help="Force one batch when it fits context limits")
    parser.add_argument("--rolling-context", type=int, default=2, help="Use last N translated lines as rolling context")
    parser.add_argument(
        "--format-mode",
        choices=["auto", "json", "schema"],
        default="auto",
        help="Structured output mode: auto=json fast path + schema retry, json=always json, schema=always schema",
    )
    parser.add_argument(
        "--minify-json",
        dest="minify_json",
        action="store_true",
        default=True,
        help="Minify JSON payload embedded in prompts (default: on)",
    )
    parser.add_argument(
        "--no-minify-json",
        dest="minify_json",
        action="store_false",
        help="Disable minified JSON payloads in prompts",
    )
    parser.add_argument("--bench", action="store_true", help="Enable detailed per-call bench logging")
    parser.add_argument("--self-test", action="store_true", help="Run internal self-tests and exit")
    args = parser.parse_args()
    set_runtime_flags(args.format_mode, args.minify_json, args.bench)

    if args.self_test:
        self_test_ass_repair_snippet()
        self_test_hybrid_pipeline()
        cprint(console, "Self-test OK", "bold green")
        return 0

    if args.batch and args.interactive:
        print("Cannot combine --batch with --interactive", file=sys.stderr)
        return 2
    if args.batch and args.out_path:
        print("--out is not supported in --batch mode", file=sys.stderr)
        return 2

    in_path = None
    out_path = None
    if not args.batch:
        if args.interactive or not args.in_path:
            try:
                in_path, out_path, args.target, args.limit, skip_summary, model = interactive_flow(
                    args, console
                )
            except RuntimeError as exc:
                print(str(exc), file=sys.stderr)
                return 2
            if skip_summary:
                args.skip_summary = True
            if model:
                args.model = model
        else:
            in_path = resolve_input_path(args.in_path)
            if not in_path.exists():
                print(f"Input not found: {in_path}", file=sys.stderr)
                return 2
            out_path = resolve_output_path(args.out_path) if args.out_path else build_output_path(in_path, args.target)

    apply_fast_profile(args, console)
    set_runtime_flags(args.format_mode, args.minify_json, args.bench)
    if args.bench:
        cprint(
            console,
            (
                f"Bench mode ON | format_mode={resolve_format_mode()} | "
                f"minify_json={'on' if MINIFY_JSON_PROMPTS else 'off'}"
            ),
            "bold cyan",
        )

    client = OllamaClient(args.host, args.model, args.timeout, args.keep_alive)

    if args.batch:
        files = collect_batch_inputs(args.in_path, args.target)
        if not files:
            cprint(console, "No subtitle files found for batch translation.", "yellow")
            return 0

        ok = 0
        skipped = 0
        failed = 0
        for file_path in files:
            out_file = build_output_path(file_path, args.target)
            if out_file.exists() and not args.overwrite:
                skipped += 1
                cprint(console, f"Skip (exists): {out_file.name}", "yellow")
                continue

            cprint(console, f"\n=== Translating: {file_path.name} ===", "bold cyan")
            code = translate_single_file(client, console, args, file_path, out_file)
            if code == 0:
                ok += 1
            else:
                failed += 1

        cprint(
            console,
            f"\nBatch summary -> ok: {ok}, skipped: {skipped}, failed: {failed}",
            "bold cyan" if failed == 0 else "bold yellow",
        )
        return 0 if failed == 0 else 1

    return translate_single_file(client, console, args, in_path, out_path)


if __name__ == "__main__":
    raise SystemExit(main())
