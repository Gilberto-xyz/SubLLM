#!/usr/bin/env python3
"""Translate .ASS or .SRT subtitles using a local Ollama model.

Workflow:
1) Read subtitles and build a brief summary for context.
2) Translate lines (SRT) or dialogue text segments (ASS), preserving tags.
"""
from __future__ import annotations

import argparse
from collections import defaultdict, OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from difflib import SequenceMatcher
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
    from rich.columns import Columns
    from rich.panel import Panel
    from rich.table import Table
    from rich import box
    from rich.progress import (
        BarColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


ASS_TAG_RE = re.compile(r"(\{[^}]*\}|\\N|\\n|\\h)")
ASS_STRIP_RE = re.compile(r"\{[^}]*\}")
ASS_VECTOR_TAG_RE = re.compile(r"\\p([1-9]\d*)")
ASS_KARAOKE_TAG_RE = re.compile(r"\\k[fo]?\d+", flags=re.IGNORECASE)
ASS_DRAWING_LINE_RE = re.compile(r"^[mlbspnc\d\s\.,+\-]+$", flags=re.IGNORECASE)
ASS_WORD_RE = re.compile(r"[A-Za-z']+")
PLACEHOLDER_TOKEN_RE = re.compile(r"__TAG_\d+__")
LEGACY_PLACEHOLDER_RE = re.compile(r"__ASS_TAG_\d+__")
BROKEN_PLACEHOLDER_RE = re.compile(r"__TAG_(?!\d+__)")
NONSPACE_TOKEN_RE = re.compile(r"\S+")
LABEL_PREFIX_RE = re.compile(
    r"^(?P<head>\s*(?:__TAG_\d+__\s*)*)(?P<label>ASS|SRT|SUBS?|CAPTION|DIALOGUE)\s*(?::|-)\s*",
    flags=re.IGNORECASE,
)
SPANISH_MARKER_RE = re.compile(r"[áéíóúñüÁÉÍÓÚÑÜ]")
CREDITS_OVERRIDE_RE = re.compile(r"\bBrought to you by\s*\[[^\]]+\]", flags=re.IGNORECASE)
SRT_WATERMARK_TOKEN_RE = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9_.-]{4,}")
FILENAME_LANG_HINTS = (
    ("English", re.compile(r"(?:^|[_.\-\s])(eng|english|ingles|ingl[eé]s)(?:$|[_.\-\s])", re.IGNORECASE)),
    ("Spanish", re.compile(r"(?:^|[_.\-\s])(spa|spanish|espanol|español)(?:$|[_.\-\s])", re.IGNORECASE)),
)
HONORIFICS = {"mr", "mrs", "ms", "miss", "sir", "ma'am", "madam"}
SUMMARY_MARKERS = (
    "summary",
    "resumen",
    "en esta escena",
    "en este episodio",
    "a lo largo de",
)
EN_STOPWORDS = {
    "the", "and", "i", "you", "your", "for", "with", "that", "this", "was", "are",
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
ASS_MAX_BATCH_ITEMS = 48
ONE_SHOT_MAX_BATCH_ITEMS = 512
DEFAULT_CTX_TOKENS = 4096
DEFAULT_PREDICT_TOKENS = 256
PROMPT_OVERHEAD_TOKENS = 640   # margen conservador para instrucciones/contexto/JSON
DEFAULT_OUT_FRACTION = 0.60    # si num_predict no viene definido, reservar ~60% ctx para output
DEFAULT_IN_OUT_RATIO = 1.15    # salida esperada ~= 1.15x entrada (traduccion subtitulos)
DEFAULT_OUT_FALLBACK_TOKENS = 192
TRANSLATION_SUMMARY_MAX = 320
TRANSLATION_TONE_MAX = 180
ROLLING_CONTEXT_MAX_CHARS = 600
FAST_SPLIT_MAX_DEPTH = 1
FAST_REPAIR_RATIO = 0.04
ADAPTIVE_BATCH_MIN_ITEMS = 8
ADAPTIVE_BATCH_REDUCE_FACTOR = 0.75
OLLAMA_RETRY_ATTEMPTS = 3
OLLAMA_RETRY_BACKOFF_SECONDS = 1.0
SUMMARY_MAX_CHUNKS = 10
SUMMARY_MAX_LINES = 1200
SUMMARY_DUP_LIMIT = 2
SRT_REPAIR_REASONS = {"empty_output", "label_prefix", "unchanged", "language_leak", "very_suspicious"}
SRT_LINEWISE_REPAIR_REASONS = {"empty_output", "label_prefix", "language_leak", "very_suspicious"}
SRT_ADAPTIVE_REDUCE_FAIL_RATE = 0.35
SRT_ADAPTIVE_REDUCE_SPLIT_FAIL_RATE = 0.15
ASS_ADAPTIVE_REDUCE_FAIL_RATE = 0.35
ASS_ADAPTIVE_REDUCE_SPLIT_FAIL_RATE = 0.15
T = TypeVar("T")
FORMAT_MODE = "auto"
MINIFY_JSON_PROMPTS = True
BENCH_MODE = False
AUTO_JSON_DISABLED = False
AUTO_JSON_ATTEMPTS = 0
AUTO_JSON_FAILS = 0
BATCH_TRANSLATION_CACHE_MAX = 200000
BATCH_TRANSLATION_CACHE = OrderedDict()
RUNTIME_DEFAULT_CTX_TOKENS = DEFAULT_CTX_TOKENS
EN_SIGNAL_TOKENS = {
    "thanks", "sorry", "please", "yes", "no", "hello", "hi", "bye", "goodbye", "good", "bad",
    "okay", "ok", "yeah", "yep", "nope", "love", "hate", "help", "wait", "stop", "go", "come",
    "look", "listen", "right", "left", "damn", "shit", "fuck", "mom", "dad", "bro", "sis",
}
EN_AMBIGUOUS_SIGNAL_TOKENS = {"no", "ok", "okay", "right", "left", "go", "come", "wait", "stop"}


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


def subtitle_type_label_and_style(path: Path) -> Tuple[str, str]:
    ext = path.suffix.lower()
    if ext == ".srt":
        return "SRT", "bold green"
    if ext == ".ass":
        return "ASS", "bold cyan"
    return ext.lstrip(".").upper() or "FILE", "white"


def show_app_header(console) -> None:
    title = "SubLLM Traductor de Subtitulos"
    subtitle = "ASS/SRT con Ollama local"
    if RICH_AVAILABLE:
        panel = Panel(
            f"[bold cyan]{title}[/bold cyan]\n[dim]{subtitle}[/dim]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 2),
        )
        console.print(panel)
    else:
        console.print(f"{title} - {subtitle}")


def print_subtitle_option(console, idx: int, path: Path) -> None:
    label, style = subtitle_type_label_and_style(path)
    if RICH_AVAILABLE:
        console.print(f"  {idx}) [{style}]{path.name}[/{style}] [dim]({label})[/dim]")
    else:
        console.print(f"  {idx}) {path.name} ({label})")


def print_subtitle_options_table(console, files: List[Path]) -> None:
    if not RICH_AVAILABLE:
        for i, f in enumerate(files, 1):
            print_subtitle_option(console, i, f)
        return
    table = Table(
        title="Archivos de subtitulos detectados",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
    )
    table.add_column("#", justify="right", style="bold white", no_wrap=True)
    table.add_column("Tipo", justify="center", no_wrap=True)
    table.add_column("Archivo", overflow="fold")
    for i, f in enumerate(files, 1):
        label, style = subtitle_type_label_and_style(f)
        table.add_row(str(i), f"[{style}]{label}[/{style}]", f.name)
    console.print(table)


def show_metrics_cards(
    console,
    metrics: List[Tuple[str, str, str | None]],
    title: str | None = None,
    columns: int = 3,
) -> None:
    if not metrics:
        return
    if not RICH_AVAILABLE:
        if title:
            cprint(console, title, "bold cyan")
        for label, value, hint in metrics:
            line = f"- {label}: {value}"
            if hint:
                line += f" ({hint})"
            console.print(line)
        return
    if title:
        cprint(console, title, "bold cyan")
    cards = []
    for label, value, hint in metrics:
        body = [f"[bold cyan]{label}[/bold cyan]", f"[bold white]{value}[/bold white]"]
        if hint:
            body.append(f"[dim]{hint}[/dim]")
        cards.append(
            Panel(
                "\n".join(body),
                box=box.ROUNDED,
                border_style="bright_black",
                padding=(0, 1),
                expand=True,
            )
        )
    chunk = max(1, int(columns))
    for idx in range(0, len(cards), chunk):
        console.print(Columns(cards[idx : idx + chunk], expand=True, equal=True))


def format_duration_compact(seconds: float) -> str:
    safe = max(0.0, float(seconds or 0.0))
    rounded = int(round(safe))
    hours, rem = divmod(rounded, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    if minutes > 0:
        return f"{minutes}m {secs}s"
    return f"{safe:.1f}s"


def print_multi_file_final_summary(
    console,
    *,
    title: str,
    selected_total: int,
    to_process: int,
    skipped: int,
    ok: int,
    failed: int,
    file_results: List[dict],
) -> None:
    processed = len(file_results)
    ok_results = [row for row in file_results if int(row.get("code", 1)) == 0]
    total_elapsed_sum = sum(float(row.get("total_elapsed", 0.0) or 0.0) for row in ok_results)
    translate_elapsed_sum = sum(float(row.get("translate_elapsed", 0.0) or 0.0) for row in ok_results)
    translated_blocks_sum = sum(int(row.get("translated_blocks", 0) or 0) for row in ok_results)
    cache_hits_sum = sum(int(row.get("cache_hits", 0) or 0) for row in ok_results)
    cache_misses_sum = sum(int(row.get("cache_misses", 0) or 0) for row in ok_results)
    cache_lookups_sum = cache_hits_sum + cache_misses_sum
    cache_rate_text = "-"
    if cache_lookups_sum > 0:
        cache_rate_text = f"{(cache_hits_sum / float(cache_lookups_sum)) * 100.0:.1f}%"

    show_metrics_cards(
        console,
        [
            ("Seleccionados", str(selected_total), None),
            ("A procesar", str(to_process), None),
            ("Procesados", str(processed), None),
            ("OK", str(ok), None),
            ("Omitidos", str(skipped), None),
            ("Fallidos", str(failed), None),
            ("Bloques traducidos", str(translated_blocks_sum), None),
            ("Tiempo traduciendo", f"{translate_elapsed_sum:.1f}s", format_duration_compact(translate_elapsed_sum)),
            ("Tiempo total acumulado", f"{total_elapsed_sum:.1f}s", format_duration_compact(total_elapsed_sum)),
            ("Cache hit global", cache_rate_text, f"aciertos={cache_hits_sum}, no_en_cache={cache_misses_sum}"),
        ],
        title=title,
        columns=3,
    )

    if not file_results:
        return

    if not RICH_AVAILABLE:
        console.print("Detalle por archivo:")
        for row in file_results:
            file_name = row.get("file_name", "?")
            status = "OK" if int(row.get("code", 1)) == 0 else f"ERROR({row.get('code', 1)})"
            blocks = row.get("translated_blocks")
            blocks_text = str(blocks) if blocks is not None else "-"
            tr_sec = row.get("translate_elapsed")
            tr_text = f"{float(tr_sec):.1f}s" if isinstance(tr_sec, (int, float)) and tr_sec > 0 else "-"
            tot_sec = row.get("total_elapsed")
            tot_text = f"{float(tot_sec):.1f}s" if isinstance(tot_sec, (int, float)) and tot_sec > 0 else "-"
            cache_hits = int(row.get("cache_hits", 0) or 0)
            cache_misses = int(row.get("cache_misses", 0) or 0)
            lookups = cache_hits + cache_misses
            if lookups > 0:
                cache_text = f"{cache_hits}/{lookups} ({(cache_hits / float(lookups)) * 100.0:.1f}%)"
            else:
                cache_text = "-"
            console.print(
                f"- {file_name} | estado={status} | bloques={blocks_text} | trad={tr_text} | total={tot_text} | cache={cache_text}"
            )
        return

    table = Table(
        title="Detalle por archivo",
        box=box.SIMPLE_HEAVY,
        header_style="bold cyan",
        show_lines=False,
        expand=True,
    )
    table.add_column("Archivo", overflow="fold")
    table.add_column("Estado", justify="center", no_wrap=True)
    table.add_column("Bloques", justify="right", no_wrap=True)
    table.add_column("Trad.", justify="right", no_wrap=True)
    table.add_column("Total", justify="right", no_wrap=True)
    table.add_column("Cache", justify="right", no_wrap=True)
    for row in file_results:
        file_name = row.get("file_name", "?")
        code = int(row.get("code", 1))
        status = "[bold green]OK[/bold green]" if code == 0 else f"[bold red]ERROR({code})[/bold red]"
        blocks = row.get("translated_blocks")
        blocks_text = str(blocks) if blocks is not None else "-"
        tr_sec = row.get("translate_elapsed")
        tr_text = f"{float(tr_sec):.1f}s" if isinstance(tr_sec, (int, float)) and tr_sec > 0 else "-"
        tot_sec = row.get("total_elapsed")
        tot_text = f"{float(tot_sec):.1f}s" if isinstance(tot_sec, (int, float)) and tot_sec > 0 else "-"
        cache_hits = int(row.get("cache_hits", 0) or 0)
        cache_misses = int(row.get("cache_misses", 0) or 0)
        lookups = cache_hits + cache_misses
        if lookups > 0:
            cache_text = f"{cache_hits}/{lookups} ({(cache_hits / float(lookups)) * 100.0:.1f}%)"
        else:
            cache_text = "-"
        table.add_row(file_name, status, blocks_text, tr_text, tot_text, cache_text)
    console.print(table)


def show_execution_roadmap(
    console,
    *,
    include_summary: bool,
    multi_file: bool,
) -> None:
    if not RICH_AVAILABLE:
        steps = ["Analizar entradas"]
        if include_summary:
            steps.extend(["Resumen", "Guia de tono"])
        steps.extend(["Traduccion", "Escritura de salida"])
        if multi_file:
            steps.append("Resumen global")
        cprint(console, "Plan de ejecucion:", "bold cyan")
        for i, step in enumerate(steps, 1):
            console.print(f"  {i}. {step}")
        return

    table = Table(
        show_header=True,
        header_style="bold cyan",
        box=box.SIMPLE_HEAVY,
        expand=True,
        row_styles=["", "dim"],
    )
    table.add_column("#", justify="right", width=3)
    table.add_column("Estado", width=13)
    table.add_column("Paso", min_width=24)
    table.add_column("Detalle", min_width=24)

    steps: List[Tuple[str, str, str]] = [
        ("PENDIENTE", "Analizar entradas", "Validacion de archivos y parametros"),
    ]
    if include_summary:
        steps.append(("PENDIENTE", "Resumen", "Contexto condensado para consistencia"))
        steps.append(("PENDIENTE", "Guia de tono", "Reglas de estilo para traducir"))
    steps.append(("ACTIVO", "Traduccion", "Lotes adaptativos + cache"))
    steps.append(("PENDIENTE", "Escritura de salida", "Persistir subtitulos traducidos"))
    if multi_file:
        steps.append(("PENDIENTE", "Resumen global", "OK/omitidos/fallidos"))

    for idx, (state, step, detail) in enumerate(steps, 1):
        state_style = "bold yellow" if state == "ACTIVO" else "bright_black"
        table.add_row(str(idx), f"[{state_style}]{state}[/{state_style}]", step, detail)

    console.print(
        Panel(
            table,
            title="Roadmap",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(0, 1),
            expand=True,
        )
    )


def progress_bar(console):
    if RICH_AVAILABLE:
        return Progress(
            SpinnerColumn(),
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
        )
    return DummyProgress()


class GlobalProgressTracker:
    STAGE_LABELS = {
        "summary": "Resumen",
        "tone_guide": "Guia de tono",
        "translate": "Traduccion",
    }

    def __init__(self, console, total_files: int, skipped: int = 0) -> None:
        self.console = console
        self.total_files = max(1, int(total_files or 1))
        self.ok = 0
        self.skipped = max(0, int(skipped or 0))
        self.failed = 0
        self._enabled = bool(RICH_AVAILABLE)
        self._progress = None
        self._global_task_id = None
        self._file_task_id = None
        self._stage_task_id = None
        self._stage_total = 1
        self._stage_completed = 0

    def __enter__(self):
        if not self._enabled:
            return self
        self._progress = Progress(
            TextColumn("{task.description}"),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            TextColumn("{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=self.console,
            transient=False,
        )
        self._progress.start()
        initial_done = min(self.total_files, self.skipped)
        self._global_task_id = self._progress.add_task(
            self._global_desc(),
            total=self.total_files,
            completed=initial_done,
        )
        self._file_task_id = self._progress.add_task("Archivo: esperando...", total=1, completed=0)
        self._stage_task_id = self._progress.add_task("Etapa: esperando...", total=1, completed=0)
        return self

    def __exit__(self, exc_type, exc, tb):
        if self._progress is not None:
            self._progress.stop()
        return False

    def _global_desc(self) -> str:
        return f"General | ok={self.ok}, omitidos={self.skipped}, fallidos={self.failed}"

    def _is_ready(self) -> bool:
        return bool(self._enabled and self._progress is not None)

    def start_file(self, file_path: Path, file_index: int, total_files: int, include_summary: bool) -> None:
        if not self._is_ready():
            return
        steps_total = 3 if include_summary else 1
        name = file_path.name
        if len(name) > 80:
            name = name[:77] + "..."
        self._progress.update(
            self._file_task_id,
            description=f"Archivo {file_index}/{total_files}: {name}",
            total=steps_total,
            completed=0,
        )
        self._progress.update(
            self._stage_task_id,
            description="Etapa: iniciando...",
            total=1,
            completed=0,
        )
        self._stage_total = 1
        self._stage_completed = 0

    def stage_start(self, stage_key: str, total: int) -> None:
        if not self._is_ready():
            return
        label = self.STAGE_LABELS.get(stage_key, stage_key)
        self._progress.update(
            self._stage_task_id,
            description=f"Etapa: {label}",
            total=max(1, int(total or 1)),
            completed=0,
        )
        self._stage_total = max(1, int(total or 1))
        self._stage_completed = 0

    def stage_advance(self, amount: int = 1) -> None:
        if not self._is_ready():
            return
        safe_amount = max(0, int(amount))
        self._stage_completed = min(self._stage_total, self._stage_completed + safe_amount)
        self._progress.update(self._stage_task_id, completed=self._stage_completed)

    def stage_done(self) -> None:
        if not self._is_ready():
            return
        if self._stage_completed < self._stage_total:
            self._progress.update(self._stage_task_id, completed=self._stage_total)
            self._stage_completed = self._stage_total
        self._progress.advance(self._file_task_id, 1)

    def set_counts(self, ok: int, skipped: int, failed: int) -> None:
        self.ok = max(0, int(ok))
        self.skipped = max(0, int(skipped))
        self.failed = max(0, int(failed))
        if not self._is_ready():
            return
        done = min(self.total_files, self.ok + self.skipped + self.failed)
        self._progress.update(
            self._global_task_id,
            description=self._global_desc(),
            completed=done,
        )


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


def set_runtime_default_ctx_tokens(num_ctx: int | None) -> None:
    global RUNTIME_DEFAULT_CTX_TOKENS
    if num_ctx is None:
        RUNTIME_DEFAULT_CTX_TOKENS = DEFAULT_CTX_TOKENS
        return
    try:
        parsed = int(num_ctx)
    except (TypeError, ValueError):
        return
    if parsed >= 1024:
        RUNTIME_DEFAULT_CTX_TOKENS = parsed


def effective_num_ctx(options) -> int:
    opts = options or {}
    raw = opts.get("num_ctx")
    if raw is None:
        return int(RUNTIME_DEFAULT_CTX_TOKENS)
    try:
        parsed = int(raw)
    except (TypeError, ValueError):
        return int(RUNTIME_DEFAULT_CTX_TOKENS)
    return max(1024, parsed)


def resolve_format_mode(mode: str | None = None) -> str:
    selected = (mode or FORMAT_MODE or "auto").strip().lower()
    if selected not in {"auto", "json", "schema"}:
        return "auto"
    return selected


def translation_cache_key(
    mode: str,
    target_lang: str,
    source: str,
    variant: int = 0,
) -> Tuple[str, str, int, str]:
    return (mode, normalize_target_lang(target_lang), int(variant or 0), source or "")


def translation_cache_get(key: Tuple[str, str, int, str]) -> str | None:
    value = BATCH_TRANSLATION_CACHE.get(key)
    if value is not None:
        BATCH_TRANSLATION_CACHE.move_to_end(key)
    return value


def translation_cache_put(key: Tuple[str, str, int, str], translated: str) -> None:
    if not translated:
        return
    BATCH_TRANSLATION_CACHE[key] = translated
    BATCH_TRANSLATION_CACHE.move_to_end(key)
    if len(BATCH_TRANSLATION_CACHE) > BATCH_TRANSLATION_CACHE_MAX:
        BATCH_TRANSLATION_CACHE.popitem(last=False)


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

    cprint(console, "Desglose de tiempos:", "bold cyan")
    cprint(console, f"- resumen: {summary_elapsed:.1f}s", "cyan")
    cprint(console, f"- guia de tono: {tone_elapsed:.1f}s", "cyan")
    cprint(console, f"- nucleo de traduccion: {translate_core_elapsed:.1f}s", "cyan")
    cprint(console, f"- reintentos (LLM): {retry_chat_elapsed:.1f}s", "cyan")
    if other_elapsed >= 0.1:
        cprint(console, f"- otros: {other_elapsed:.1f}s", "cyan")

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
                "Contadores de reintentos: "
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
    fast_trunc_batches = RUNTIME_METRICS.counters.get("retry.fast_budget.truncated_batches", 0)
    if fast_trunc_batches:
        fast_trunc_items = RUNTIME_METRICS.counters.get("retry.fast_budget.truncated_items", 0)
        cprint(
            console,
            f"Presupuesto rapido de reintentos: lotes_recortados={fast_trunc_batches}, items_omitidos={fast_trunc_items}",
            "cyan",
        )
    batch_calls = RUNTIME_METRICS.counters.get("translate.json_batch.calls", 0)
    top_batches = RUNTIME_METRICS.counters.get("translate.top_level_batches", 0)
    if top_batches:
        top_items = RUNTIME_METRICS.counters.get("translate.top_level_batch_items", 0)
        cprint(
            console,
            f"Lotes de primer nivel: {top_batches} (items_promedio={top_items / float(top_batches):.1f})",
            "cyan",
        )
    if batch_calls:
        batch_items = RUNTIME_METRICS.counters.get("translate.json_batch.items", 0)
        batch_chars = RUNTIME_METRICS.counters.get("translate.json_batch.input_chars", 0)
        avg_items = batch_items / float(batch_calls)
        avg_chars = batch_chars / float(batch_calls)
        cprint(
            console,
            f"Estadisticas de lotes: llamadas={batch_calls}, items_promedio={avg_items:.1f}, chars_entrada_promedio={avg_chars:.0f}",
            "cyan",
        )
    cache_hits = RUNTIME_METRICS.counters.get("translate.cache.hits", 0)
    cache_misses = RUNTIME_METRICS.counters.get("translate.cache.misses", 0)
    cache_writes = RUNTIME_METRICS.counters.get("translate.cache.writes", 0)
    cache_lookups = cache_hits + cache_misses
    if cache_lookups:
        hit_rate = (cache_hits / float(cache_lookups)) * 100.0
        cprint(
            console,
            (
                "Cache de traduccion: "
                f"consultas={cache_lookups}, aciertos={cache_hits}, no_en_cache={cache_misses}, "
                f"escrituras={cache_writes}, tasa_acierto={hit_rate:.1f}%"
            ),
            "cyan",
        )
    schema_retries = RUNTIME_METRICS.counters.get("format.schema_retry.attempts", 0)
    if schema_retries:
        cprint(console, f"Fallbacks de formato: schema_retry={schema_retries}", "cyan")
    split_recursions = RUNTIME_METRICS.counters.get("translate.split.recursions", 0)
    split_cutoff = RUNTIME_METRICS.counters.get("translate.split.depth_cutoff", 0)
    if split_recursions or split_cutoff:
        cprint(
            console,
            f"Estadisticas de split: recursiones={split_recursions}, corte_profundidad={split_cutoff}",
            "cyan",
        )
    reason_entries = [
        (name[len("status.reason.") :], count)
        for name, count in RUNTIME_METRICS.counters.items()
        if name.startswith("status.reason.") and count > 0
    ]
    if reason_entries:
        reason_entries.sort(key=lambda pair: pair[1], reverse=True)
        top = ", ".join(f"{reason}={count}" for reason, count in reason_entries[:6])
        cprint(console, f"Motivos mas marcados: {top}", "cyan")
    if resolve_format_mode() == "auto":
        cprint(
            console,
            (
                f"Estadisticas auto-formato: intentos_json={AUTO_JSON_ATTEMPTS}, "
                f"fallos_json={AUTO_JSON_FAILS}, json_deshabilitado={AUTO_JSON_DISABLED}"
            ),
            "cyan",
        )
    ollama_calls = RUNTIME_METRICS.counters.get("ollama.calls", 0)
    if ollama_calls:
        avg_ollama = RUNTIME_METRICS.seconds.get("ollama.chat", 0.0) / float(ollama_calls)
        avg_prompt_chars = RUNTIME_METRICS.counters.get("ollama.prompt_chars", 0) / float(ollama_calls)
        cprint(
            console,
            f"Llamadas a Ollama: {ollama_calls} (promedio={avg_ollama:.2f}s, chars_prompt_promedio={avg_prompt_chars:.0f})",
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
        self._detected_num_ctx: int | None = None

    def detect_model_num_ctx(self) -> int | None:
        if self._detected_num_ctx is not None:
            return self._detected_num_ctx
        payload = {"model": self.model}
        url = f"{self.host}/api/show"
        data = dump_json(payload, minify=True).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=min(10, self.timeout)) as resp:
                body = resp.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
        except Exception:
            return None

        candidates: List[int] = []
        model_info = parsed.get("model_info")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                if "context_length" not in str(key):
                    continue
                try:
                    parsed_val = int(value)
                except (TypeError, ValueError):
                    continue
                if parsed_val >= 1024:
                    candidates.append(parsed_val)

        parameters = parsed.get("parameters")
        if isinstance(parameters, str):
            for match in re.finditer(r"(?:^|\n)\s*num_ctx\s+(\d+)", parameters):
                try:
                    parsed_val = int(match.group(1))
                except (TypeError, ValueError):
                    continue
                if parsed_val >= 1024:
                    candidates.append(parsed_val)

        if not candidates:
            return None
        self._detected_num_ctx = max(candidates)
        return self._detected_num_ctx

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
        attempts = max(1, OLLAMA_RETRY_ATTEMPTS)
        body = ""
        last_exc = None
        for attempt in range(1, attempts + 1):
            try:
                with request.urlopen(req, timeout=self.timeout) as resp:
                    body = resp.read().decode("utf-8", errors="replace")
                last_exc = None
                break
            except socket.timeout as exc:
                last_exc = OllamaTimeoutError(
                    f"Ollama request timed out after {self.timeout}s "
                    f"(attempt {attempt}/{attempts})"
                )
            except TimeoutError as exc:
                last_exc = OllamaTimeoutError(
                    f"Ollama request timed out after {self.timeout}s "
                    f"(attempt {attempt}/{attempts})"
                )
            except error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
            except error.URLError as exc:
                reason = exc.reason
                if isinstance(reason, (TimeoutError, socket.timeout)):
                    last_exc = OllamaTimeoutError(
                        f"Ollama request timed out after {self.timeout}s "
                        f"(attempt {attempt}/{attempts})"
                    )
                elif isinstance(reason, ConnectionRefusedError):
                    last_exc = RuntimeError(
                        f"Cannot reach Ollama at {self.host} "
                        f"(connection refused, attempt {attempt}/{attempts}). Is it running?"
                    )
                else:
                    raise RuntimeError(
                        f"Cannot reach Ollama at {self.host}. Is it running?"
                    ) from exc
            except ConnectionRefusedError as exc:
                last_exc = RuntimeError(
                    f"Cannot reach Ollama at {self.host} "
                    f"(connection refused, attempt {attempt}/{attempts}). Is it running?"
                )
            if attempt < attempts:
                time.sleep(OLLAMA_RETRY_BACKOFF_SECONDS * attempt)
        if last_exc is not None:
            raise last_exc
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


def evenly_sample_lines(lines: List[str], limit: int) -> List[str]:
    if limit <= 0 or len(lines) <= limit:
        return lines
    if limit == 1:
        return [lines[len(lines) // 2]]
    step = (len(lines) - 1) / float(limit - 1)
    out = []
    last_idx = -1
    for i in range(limit):
        idx = int(round(i * step))
        if idx <= last_idx:
            idx = min(len(lines) - 1, last_idx + 1)
        out.append(lines[idx])
        last_idx = idx
    return out


def prepare_summary_lines(lines: List[str], max_chars: int) -> Tuple[List[str], int]:
    cleaned = []
    seen_counts: dict[str, int] = {}
    for line in lines:
        normalized = normalize_inline_text(line)
        if not normalized:
            continue
        key = normalized.lower()
        seen = seen_counts.get(key, 0)
        if seen >= SUMMARY_DUP_LIMIT:
            continue
        seen_counts[key] = seen + 1
        cleaned.append(normalized)
    if not cleaned:
        return [], 0

    sampled = evenly_sample_lines(cleaned, SUMMARY_MAX_LINES)
    max_total_chars = max(max_chars, max_chars * SUMMARY_MAX_CHUNKS)
    total_chars = sum(len(line) + 1 for line in sampled)
    if total_chars > max_total_chars:
        target_lines = max(1, int(len(sampled) * (max_total_chars / float(total_chars))))
        sampled = evenly_sample_lines(sampled, target_lines)
        trimmed = []
        running = 0
        for line in sampled:
            line_len = len(line) + 1
            if trimmed and running + line_len > max_total_chars:
                continue
            trimmed.append(line)
            running += line_len
        sampled = trimmed or sampled[:1]
    return sampled, len(cleaned)


def summarize_subs(
    client: OllamaClient,
    lines: List[str],
    max_chars: int,
    options,
    console,
    progress_tracker: GlobalProgressTracker | None = None,
) -> str:
    if progress_tracker is not None:
        progress_tracker.stage_start("summary", 1)
    if not lines:
        if progress_tracker is not None:
            progress_tracker.stage_advance(1)
            progress_tracker.stage_done()
        return ""
    summary_lines, cleaned_count = prepare_summary_lines(lines, max_chars)
    if not summary_lines:
        if progress_tracker is not None:
            progress_tracker.stage_advance(1)
            progress_tracker.stage_done()
        return ""
    if len(summary_lines) < cleaned_count:
        cprint(
            console,
            f"Corpus de resumen reducido: {cleaned_count} -> {len(summary_lines)} linea(s).",
            "yellow",
        )
    with RUNTIME_METRICS.timed("summary.total"):
        chunks = build_chunks(summary_lines, max_chars)
        RUNTIME_METRICS.bump("summary.chunks", len(chunks))
        summaries = []
        system_msg = (
            "You summarize subtitles. Write a concise summary in Spanish. "
            "Keep it short and factual."
        )
        merge_units = 1 if len(chunks) > 1 else 0
        total_units = len(chunks) + merge_units
        if progress_tracker is not None:
            progress_tracker.stage_start("summary", total_units)
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
                progress_tracker.stage_advance(1)
        else:
            with progress_bar(console) as progress:
                task_id = progress.add_task("Resumen", total=len(chunks))
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
            if progress_tracker is not None:
                progress_tracker.stage_done()
            return summaries[0]
        user_msg = (
            "Combine these partial summaries into a single short summary in Spanish. "
            "Avoid repetition.\n\n" + "\n\n".join(summaries)
        )
        with RUNTIME_METRICS.timed("summary.chat.merge"):
            merged = client.chat(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options=options,
            ).strip()
        if progress_tracker is not None:
            progress_tracker.stage_advance(1)
            progress_tracker.stage_done()
        return merged


def build_tone_guide(
    client: OllamaClient,
    summary: str,
    sample_lines: List[str],
    options,
    console,
    progress_tracker: GlobalProgressTracker | None = None,
) -> str:
    if progress_tracker is not None:
        progress_tracker.stage_start("tone_guide", 1)
    if not summary:
        if progress_tracker is not None:
            progress_tracker.stage_advance(1)
            progress_tracker.stage_done()
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
    cprint(console, "Construyendo guia de tono...", "bold cyan")
    with RUNTIME_METRICS.timed("tone_guide.total"):
        with RUNTIME_METRICS.timed("tone_guide.chat"):
            content = client.chat(
                [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                options=options,
            )
    if progress_tracker is not None:
        progress_tracker.stage_advance(1)
        progress_tracker.stage_done()
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


def english_signal_score(text: str) -> int:
    tokens = ascii_word_tokens(text)
    if not tokens:
        return 0
    stop_hits = sum(1 for token in tokens if token in EN_STOPWORDS)
    honor_hits = sum(1 for token in tokens if token in HONORIFICS)
    contraction_hits = sum(1 for token in tokens if "'" in token)
    strong_signal_hits = sum(
        1
        for token in tokens
        if token in EN_SIGNAL_TOKENS and token not in EN_AMBIGUOUS_SIGNAL_TOKENS
    )
    weak_signal_hits = sum(1 for token in tokens if token in EN_SIGNAL_TOKENS)
    score = (
        stop_hits * 2
        + honor_hits * 2
        + contraction_hits * 2
        + strong_signal_hits * 2
        + weak_signal_hits
    )
    if SPANISH_MARKER_RE.search(text or ""):
        score -= 2
    return max(0, score)


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
    src_en_score = english_signal_score(source)

    if mode == "es":
        # A standalone "I" in Spanish output is a strong incomplete-translation signal.
        if "i" in dst_tokens and src_en_score >= 3:
            return True
        if src_en_score < 3:
            # For non-English source lines (romaji, names, effects), avoid expensive false positives.
            return en_hits >= 4 and not has_spanish_signal and len(dst_tokens) >= 4
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
    return english_signal_score(src) >= 2


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


def needs_llm_repair_for_reasons(status: dict, repair_reasons: set[str]) -> bool:
    reasons = status.get("reasons") if isinstance(status, dict) else None
    if not reasons:
        return False
    return any(reason in repair_reasons for reason in reasons)


def effective_repair_reasons_for_item(
    item: dict,
    base_reasons: set[str],
    target_lang: str,
) -> set[str]:
    reasons = set(base_reasons)
    if normalize_target_lang(target_lang) != "es":
        return reasons
    source = str(
        item.get("original_text_field")
        or item.get("text_field")
        or item.get("protected_text")
        or ""
    )
    item_kind = str(item.get("kind") or "")
    source_plain = ass_plain_text(source) if item_kind.startswith("ass") else source
    if english_signal_score(source_plain) < 2:
        reasons.discard("language_leak")
        reasons.discard("very_suspicious")
        reasons.discard("unchanged")
    return reasons


def item_needs_llm_repair(item: dict, base_reasons: set[str], target_lang: str) -> bool:
    status = item.get("status", {})
    reasons = effective_repair_reasons_for_item(item, base_reasons, target_lang)
    return needs_llm_repair_for_reasons(status, reasons)


def fast_repair_budget(batch_len: int) -> int:
    return max(2, int(math.ceil(FAST_REPAIR_RATIO * max(1, batch_len))))


def trim_failed_items_for_fast_budget(failed_items: List[dict], budget: int) -> List[dict]:
    if budget <= 0 or len(failed_items) <= budget:
        return failed_items
    ranked = sorted(
        failed_items,
        key=lambda item: int(item.get("status", {}).get("score", 0)),
        reverse=True,
    )
    RUNTIME_METRICS.bump("retry.fast_budget.truncated_batches")
    RUNTIME_METRICS.bump("retry.fast_budget.truncated_items", len(failed_items) - budget)
    return ranked[:budget]


def bump_status_reason_counters(status: dict) -> None:
    reasons = status.get("reasons") if isinstance(status, dict) else None
    if not reasons:
        return
    for reason in reasons:
        RUNTIME_METRICS.bump(f"status.reason.{reason}")


def should_translate(segment: str) -> bool:
    return bool(re.search(r"[A-Za-z]", segment))


def normalize_inline_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\n", " ").strip())


def is_summary_echo(candidate: str, summary: str) -> bool:
    cand = normalize_inline_text(candidate).lower()
    summ = normalize_inline_text(summary).lower()
    if not cand or not summ:
        return False
    if cand == summ:
        return True

    if len(cand) >= 80 and len(summ) >= 80:
        prefix_len = min(96, len(cand), len(summ))
        if prefix_len >= 48 and cand[:prefix_len] == summ[:prefix_len]:
            return True

    if len(cand) >= 64 and len(summ) >= 64:
        shorter = min(len(cand), len(summ))
        longer = max(len(cand), len(summ))
        if shorter / float(longer) >= 0.85 and (cand in summ or summ in cand):
            return True

    if len(cand) >= 48 and len(summ) >= 48:
        if SequenceMatcher(None, cand, summ).ratio() >= 0.72:
            return True

    cand_tokens = [token for token in ascii_word_tokens(cand) if len(token) >= 4]
    summ_tokens = [token for token in ascii_word_tokens(summ) if len(token) >= 4]
    if len(cand_tokens) >= 6 and len(summ_tokens) >= 8:
        cand_set = set(cand_tokens)
        summ_set = set(summ_tokens)
        shared = cand_set.intersection(summ_set)
        if len(shared) >= 6:
            cand_cov = len(shared) / float(max(1, len(cand_set)))
            summ_cov = len(shared) / float(max(1, len(summ_set)))
            if cand_cov >= 0.65 and summ_cov >= 0.35:
                return True

    return False


def is_ass_drawing_payload(text_field: str, plain_text: str) -> bool:
    source = text_field or ""
    if ASS_VECTOR_TAG_RE.search(source):
        return True
    normalized = normalize_inline_text(plain_text)
    if not normalized:
        return False
    if ASS_DRAWING_LINE_RE.fullmatch(normalized):
        return True
    tokens = normalized.split()
    if not tokens:
        return False
    draw_cmds = sum(1 for token in tokens if token.lower() in {"m", "l", "b", "s", "p", "n", "c"})
    numeric = sum(1 for token in tokens if re.fullmatch(r"[+-]?\d+(?:\.\d+)?", token))
    return draw_cmds >= 3 and numeric >= 2 and (draw_cmds + numeric) >= max(4, int(len(tokens) * 0.7))


def looks_like_low_value_ass_fragment(plain_text: str) -> bool:
    normalized = normalize_inline_text(plain_text)
    if not normalized:
        return True
    words = ASS_WORD_RE.findall(normalized.lower())
    if not words:
        return True
    letters = sum(len(word) for word in words)
    if len(words) == 1 and letters <= 3:
        return True
    if len(words) <= 2 and letters <= 4:
        return True
    if len(set(words)) == 1 and len(words) >= 2 and len(words[0]) <= 3:
        return True
    return False


def is_karaoke_syllable_payload(text_field: str) -> bool:
    if not text_field or not ASS_KARAOKE_TAG_RE.search(text_field):
        return False
    tokens = split_ass_text(text_field)
    text_parts: List[str] = []
    for token in tokens:
        if not token or token.startswith("{") or token in {"\\N", "\\n", "\\h"}:
            continue
        cleaned = token.strip()
        if cleaned:
            text_parts.append(cleaned)
    if len(text_parts) < 4:
        return False
    alpha_parts = [part for part in text_parts if ASS_WORD_RE.search(part)]
    if len(alpha_parts) < 4:
        return False
    compact_lengths = [len(re.sub(r"\s+", "", part)) for part in alpha_parts]
    if not compact_lengths:
        return False
    if max(compact_lengths) <= 3 and (sum(1 for n in compact_lengths if n <= 2) / float(len(compact_lengths))) >= 0.7:
        return True
    return False


def should_translate_ass_dialogue(text_field: str, effect: str = "", for_summary: bool = False) -> bool:
    plain = ass_plain_text(text_field)
    normalized = normalize_inline_text(plain)
    if not normalized:
        return False
    if not should_translate(normalized):
        return False
    if is_ass_drawing_payload(text_field, normalized):
        return False
    if is_karaoke_syllable_payload(text_field):
        return False
    words = ASS_WORD_RE.findall(normalized.lower())
    effect_norm = (effect or "").strip().lower()
    if effect_norm == "fx" and looks_like_low_value_ass_fragment(normalized):
        return False
    if for_summary:
        if looks_like_low_value_ass_fragment(normalized):
            return False
        if effect_norm == "fx" and len(words) < 4:
            return False
        if len(normalized) < 6:
            return False
    return True


def should_skip_srt_block_translation(text: str) -> bool:
    normalized = " ".join(part.strip() for part in (text or "").split("\n") if part.strip())
    if not normalized:
        return True
    if not should_translate(normalized):
        return True
    return SRT_WATERMARK_TOKEN_RE.fullmatch(normalized) is not None


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
    raw = text or ""
    char_est = int(math.ceil(len(raw) / 4.0))
    word_est = int(math.ceil(len(NONSPACE_TOKEN_RE.findall(raw)) * 0.90))
    # Placeholders/token-like artifacts are usually more expensive than plain text.
    placeholder_penalty = (
        len(PLACEHOLDER_TOKEN_RE.findall(raw)) * 3
        + len(LEGACY_PLACEHOLDER_RE.findall(raw)) * 3
        + len(BROKEN_PLACEHOLDER_RE.findall(raw)) * 2
    )
    newline_penalty = raw.count("\n")
    return max(1, max(char_est, word_est) + placeholder_penalty + newline_penalty)


def reduce_adaptive_batch_cap(current_cap: int) -> int:
    if current_cap <= ADAPTIVE_BATCH_MIN_ITEMS:
        return current_cap
    reduced = int(math.floor(current_cap * ADAPTIVE_BATCH_REDUCE_FACTOR))
    return max(ADAPTIVE_BATCH_MIN_ITEMS, reduced)


def translation_budget(options) -> Tuple[int, int]:
    opts = options or {}
    num_ctx = effective_num_ctx(opts)
    num_predict = int(opts.get("num_predict") or 0)

    if num_predict > 0:
        reserve_out = max(128, min(4096, num_predict))
        budget = max(256, num_ctx - reserve_out - PROMPT_OVERHEAD_TOKENS)
        return budget, reserve_out

    # Presupuesto conservador basado en ratio entrada/salida típico de traducción.
    usable = max(512, num_ctx - PROMPT_OVERHEAD_TOKENS)
    est_budget = int(math.floor((usable - DEFAULT_OUT_FALLBACK_TOKENS) / (1.0 + DEFAULT_IN_OUT_RATIO)))
    max_budget = max(256, usable - DEFAULT_OUT_FALLBACK_TOKENS)
    budget = max(256, min(max_budget, est_budget))
    reserve_out = max(DEFAULT_OUT_FALLBACK_TOKENS, usable - budget)
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


def parse_array_response_loose(content: str) -> List[str] | None:
    cleaned = content.strip()
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [str(item) for item in data]
        if isinstance(data, dict):
            for key in ("translations", "items", "data", "result", "output"):
                value = data.get(key)
                if isinstance(value, list):
                    return [str(item) for item in value]
    except json.JSONDecodeError:
        pass
    fallback = extract_json_array(content)
    if fallback is not None:
        return fallback
    cleaned = re.sub(r"```(?:json)?", "", cleaned, flags=re.IGNORECASE).replace("```", "")
    start = cleaned.find("[")
    if start != -1:
        payload = cleaned[start:]
        decoder = json.JSONDecoder()
        idx = 1
        out: List[str] = []
        while idx < len(payload):
            while idx < len(payload) and payload[idx] in " \r\n\t,":
                idx += 1
            if idx >= len(payload):
                break
            if payload[idx] == "]":
                return out
            try:
                value, next_idx = decoder.raw_decode(payload, idx)
            except json.JSONDecodeError:
                break
            out.append(str(value))
            idx = next_idx
        if out:
            return out
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
    num_ctx = effective_num_ctx(options_batch)
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
    budget, _reserve_out = translation_budget(options)
    cap = len(items) if batch_size is None or int(batch_size) <= 0 else max(1, int(batch_size))
    est_total = sum(item.get("in_tokens", estimate_tokens_from_text(item.get("protected_text", ""))) for item in items)
    if est_total <= budget and len(items) <= cap:
        return [items]

    if one_shot:
        if len(items) > cap:
            cprint(
                console,
                f"--one-shot solicitado pero batch-size={cap} < items={len(items)}; usando lotes adaptativos.",
                "yellow",
            )
        elif est_total <= budget:
            return [items]
        else:
            cprint(
                console,
                "--one-shot solicitado pero el prompt estimado no entra en num_ctx; usando lotes adaptativos.",
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
    allow_partial: bool = False,
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
    if mode in {"ass_protected", "srt_block"} and not force_temperature_zero:
        options_batch["temperature"] = 0.0
    request_format: str | dict
    used_auto_json = False
    if selected_format_mode == "schema":
        request_format = schema
    elif selected_format_mode == "auto" and (
        force_temperature_zero
        or len(texts) <= 2
        or AUTO_JSON_DISABLED
        or mode in {"ass_protected", "srt_block"}
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
    if allow_partial:
        partial = parse_array_response_loose(content)
        if partial:
            if BENCH_MODE:
                print(
                    f"[bench] partial_parse accepted size={len(partial)} expected={len(texts)}"
                )
            return partial
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
    allow_partial: bool = False,
    depth: int = 0,
    max_split_depth: int | None = None,
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
            allow_partial=allow_partial,
        )
    except OllamaTimeoutError:
        out = None
    except RuntimeError as exc:
        if "timed out" in str(exc).lower():
            out = None
        else:
            out = None

    if out is not None:
        cleaned_out = [strip_label_prefix(item) for item in out]
        if summary and cleaned_out and is_summary_echo(cleaned_out[0], summary):
            RUNTIME_METRICS.bump("translate.summary_echo_rejected")
        else:
            if len(cleaned_out) == len(texts):
                return cleaned_out
            if allow_partial:
                if len(cleaned_out) > len(texts):
                    return cleaned_out[: len(texts)]
                return cleaned_out + [None] * (len(texts) - len(cleaned_out))
    if len(texts) == 1:
        return [None]
    if max_split_depth is not None and depth >= max_split_depth:
        RUNTIME_METRICS.bump("translate.split.depth_cutoff")
        return [None] * len(texts)
    RUNTIME_METRICS.bump("translate.split.recursions")
    RUNTIME_METRICS.bump("translate.split.recursion_items", len(texts))

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
        allow_partial=allow_partial,
        depth=depth + 1,
        max_split_depth=max_split_depth,
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
        allow_partial=allow_partial,
        depth=depth + 1,
        max_split_depth=max_split_depth,
    )
    return left + right


def repair_issue_score(reasons: List[str]) -> int:
    weights = {
        "empty_output": 6,
        "placeholder_mismatch": 6,
        "unchanged": 6,
        "duplicate_output": 6,
        "short_line_overflow": 5,
        "length_mismatch": 4,
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
    # If the model leaked placeholder-like tokens in plain text, drop them before slot rebuild.
    plain = PLACEHOLDER_TOKEN_RE.sub(" ", plain)
    plain = LEGACY_PLACEHOLDER_RE.sub(" ", plain)
    plain = BROKEN_PLACEHOLDER_RE.sub(" ", plain)
    plain = re.sub(r"[ \t]+", " ", plain).strip()
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
    src_words = count_words(source_plain)
    dst_words = count_words(restored_plain)
    if src_words <= 3 and dst_words >= max(8, src_words * 4):
        reasons.append("short_line_overflow")
    elif src_words <= 3:
        sentence_marks = restored_plain.count(".") + restored_plain.count("?") + restored_plain.count("!")
        if sentence_marks >= 2 and dst_words >= 5:
            reasons.append("short_line_overflow")
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
    if "placeholder_mismatch" in reasons or "placeholder_artifacts" in reasons:
        recovered = recover_ass_candidate_placeholders(state, cleaned)
        if recovered is not None and recovered != cleaned:
            evaluation = evaluate_ass_line_candidate(state, recovered, target_lang)
            reasons = evaluation["reasons"]

    ok = len(reasons) == 0
    needs_llm_repair = any(
        reason in {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged", "language_leak", "very_suspicious", "short_line_overflow"}
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
    src_words = count_words(source_token)
    dst_words = count_words(cleaned)
    if src_words <= 3 and dst_words >= max(8, src_words * 4):
        reasons.append("short_line_overflow")
    elif src_words <= 3:
        sentence_marks = cleaned.count(".") + cleaned.count("?") + cleaned.count("!")
        if sentence_marks >= 2 and dst_words >= 5:
            reasons.append("short_line_overflow")

    ok = len(reasons) == 0
    needs_llm_repair = any(
        reason in {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged", "language_leak", "very_suspicious", "short_line_overflow"}
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
    summary = "The group is trapped in a desperate situation and plans a chaotic strategy."
    assert is_summary_echo(summary, summary)
    assert is_summary_echo(summary + " ", summary)
    paraphrase = "A group is trapped in a desperate situation and tries a chaotic strategy."
    assert is_summary_echo(paraphrase, summary)
    assert not is_summary_echo("I am fine.", summary)
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
    srt_wrap_status = scan_srt_candidate(srt_source, "Hola alli", "Spanish")
    assert "line_break_mismatch" in srt_wrap_status["reasons"]
    assert not srt_wrap_status["needs_llm_repair"]
    assert should_skip_srt_block_translation("Katmovie18")
    assert not should_skip_srt_block_translation("Whatever.")
    assert not should_translate_ass_dialogue(r"{\an5\p1}m 0 -6 l -4.5 -1.5 l -4.5 3 l 0 7.5", effect="fx")
    assert not should_translate_ass_dialogue(r"{\an5\pos(766.5,64.5)}da", effect="fx", for_summary=True)
    karaoke_line = (
        r"{\r\t(0,1230,\c&H704215&\kf123)}da{\r\t(1230,1280,\c&H704215&\kf5)}s"
        r"{\r\t(1280,1470,\c&H704215&\kf19)}so{\r\t(1470,1610,\c&H704215&\kf14)}u "
        r"{\r\t(1610,1730,\c&H704215&\kf12)}da{\r\t(1730,1790,\c&H704215&\kf6)}s"
        r"{\r\t(1790,2070,\c&H704215&\kf28)}so{\r\t(2070,2350,\c&H704215&\kf28)}u"
    )
    assert is_karaoke_syllable_payload(karaoke_line)
    assert not should_translate_ass_dialogue(karaoke_line, effect="fx")
    assert should_translate_ass_dialogue(
        "Word on the street is that policy changed right when the Council Chairman did.",
        effect="",
    )
    dedupe_input = [
        {"protected_text": "__TAG_0__Warning"},
        {"protected_text": "__TAG_0__Warning"},
        {"protected_text": "__TAG_0__Underground Student Council"},
    ]
    leaders, groups = dedupe_ass_translation_items(dedupe_input)
    assert len(leaders) == 2
    assert len(groups["__TAG_0__Warning"]) == 2
    srt_dedupe_input = [
        {"protected_text": "Hello there", "expected_line_count": 1},
        {"protected_text": "Hello there", "expected_line_count": 1},
        {"protected_text": "Hello there", "expected_line_count": 2},
    ]
    srt_leaders, srt_groups = dedupe_srt_translation_items(srt_dedupe_input)
    assert len(srt_leaders) == 2
    assert len(srt_groups[("Hello there", 1)]) == 2
    duplicate_batch = [
        {
            "original_text_field": "I saw him near the station yesterday.",
            "status": scan_srt_candidate(
                {"original_text_field": "I saw him near the station yesterday.", "expected_line_count": 1},
                "Lo vi cerca de la estacion ayer.",
                "Spanish",
            ),
        },
        {
            "original_text_field": "Please lock the front door before sleeping.",
            "status": scan_srt_candidate(
                {"original_text_field": "Please lock the front door before sleeping.", "expected_line_count": 1},
                "Lo vi cerca de la estacion ayer.",
                "Spanish",
            ),
        },
    ]
    mark_duplicate_srt_batch_outputs(duplicate_batch)
    assert "duplicate_output" in duplicate_batch[1]["status"]["reasons"]
    assert not duplicate_batch[1]["status"]["needs_llm_repair"]
    duplicate_window_batch = [
        {
            "original_text_field": "Keep the back gate locked tonight.",
            "status": scan_srt_candidate(
                {"original_text_field": "Keep the back gate locked tonight.", "expected_line_count": 1},
                "Manten cerrada la puerta trasera esta noche.",
                "Spanish",
            ),
        },
        {
            "original_text_field": "Okay.",
            "status": scan_srt_candidate(
                {"original_text_field": "Okay.", "expected_line_count": 1},
                "Vale.",
                "Spanish",
            ),
        },
        {
            "original_text_field": "Please turn off the lights before leaving.",
            "status": scan_srt_candidate(
                {"original_text_field": "Please turn off the lights before leaving.", "expected_line_count": 1},
                "Manten cerrada la puerta trasera esta noche.",
                "Spanish",
            ),
        },
    ]
    mark_duplicate_srt_batch_outputs(duplicate_window_batch)
    assert "duplicate_output" in duplicate_window_batch[2]["status"]["reasons"]
    overflow_status = scan_srt_candidate(
        {"original_text_field": "I mean,", "expected_line_count": 1},
        "No lo creo, senora. Disculpa? O sea,",
        "Spanish",
    )
    assert "short_line_overflow" in overflow_status["reasons"]
    assert not overflow_status["needs_llm_repair"]
    length_status = scan_srt_candidate(
        {"original_text_field": "it is not in academics.", "expected_line_count": 1},
        "Aun esta descubriendo en que area puede destacar realmente frente a los demas.",
        "Spanish",
    )
    assert "length_mismatch" in length_status["reasons"]
    assert not length_status["needs_llm_repair"]
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
    overflow_src = "dassou dassou"
    overflow_protected, overflow_repl = ass_protect_tags(overflow_src)
    overflow_state = {
        "text_field": overflow_src,
        "protected": overflow_protected,
        "replacements": overflow_repl,
    }
    overflow_ass_status = scan_ass_line_candidate(
        overflow_state,
        "Un grupo de personas se encuentra en una situacion desesperada, amenazados con la expulsion.",
        "Spanish",
    )
    assert "short_line_overflow" in overflow_ass_status["reasons"]
    assert overflow_ass_status["needs_llm_repair"]
    credits = "{\\an8}Brought to you by [ToonsHub]"
    assert apply_phrase_overrides(credits) == "{\\an8}Tra\u00eddo por [el_inmortus]"


def parse_ass_ir(text: str, limit: int | None) -> Tuple[List[str], List[dict]]:
    lines = text.splitlines()
    in_events = False
    format_fields = None
    text_idx = None
    effect_idx = None
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
            effect_idx = format_fields.index("Effect") if "Effect" in format_fields else None
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
        effect = fields[effect_idx] if effect_idx is not None else ""
        if not should_translate_ass_dialogue(text_field, effect=effect):
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


def dedupe_ass_translation_items(items: List[dict]) -> Tuple[List[dict], dict[str, List[dict]]]:
    groups: dict[str, List[dict]] = {}
    order: List[str] = []
    for item in items:
        key = item.get("protected_text", "")
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [item]
            order.append(key)
        else:
            bucket.append(item)
    leaders = [groups[key][0] for key in order]
    return leaders, groups


def dedupe_srt_translation_items(items: List[dict]) -> Tuple[List[dict], dict[Tuple[str, int], List[dict]]]:
    groups: dict[Tuple[str, int], List[dict]] = {}
    order: List[Tuple[str, int]] = []
    for item in items:
        key = (item.get("protected_text", ""), int(item.get("expected_line_count", 1)))
        bucket = groups.get(key)
        if bucket is None:
            groups[key] = [item]
            order.append(key)
        else:
            bucket.append(item)
    leaders = [groups[key][0] for key in order]
    return leaders, groups


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
        elif should_skip_srt_block_translation(joined):
            fixed_final_text = joined
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


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower())


def mark_duplicate_srt_batch_outputs(items: List[dict]) -> None:
    recent: List[Tuple[str, str]] = []
    for item in items:
        status = item.get("status")
        if not isinstance(status, dict):
            continue
        candidate = _normalize_ws(status.get("candidate", ""))
        source = _normalize_ws(item.get("original_text_field", ""))
        if candidate:
            for prev_candidate, prev_source in recent[-3:]:
                if candidate != prev_candidate:
                    continue
                if not source or source == prev_source:
                    continue
                if count_words(candidate) >= 3 or len(candidate) >= 20:
                    if count_words(source) >= 3 or count_words(prev_source) >= 3:
                        reasons = list(status.get("reasons") or [])
                        if "duplicate_output" not in reasons:
                            reasons.append("duplicate_output")
                            status["reasons"] = reasons
                            status["ok"] = False
                            status["needs_llm_repair"] = False
                            status["score"] = repair_issue_score(reasons)
                        break
        recent.append((candidate, source))


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
    src_words = count_words(source_text)
    dst_words = count_words(cleaned)
    if src_words <= 3 and dst_words >= max(5, src_words * 3):
        reasons.append("short_line_overflow")
    elif src_words <= 3:
        sentence_marks = cleaned.count(".") + cleaned.count("?") + cleaned.count("!")
        if sentence_marks >= 2 and dst_words >= 4:
            reasons.append("short_line_overflow")
        elif ("?" in source_text) and ("?" not in cleaned and "¿" not in cleaned) and dst_words >= 3:
            reasons.append("short_line_overflow")
    elif 4 <= src_words <= 10:
        if dst_words <= 1 or dst_words >= int(math.ceil(src_words * 1.8)) + 1:
            reasons.append("length_mismatch")

    # For SRT we can safely normalize line wraps later with enforce_line_count().
    # Treat line break drift as non-fatal to avoid expensive false-positive retries.
    needs_repair = any(
        reason in SRT_REPAIR_REASONS
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
    fast_mode: bool = False,
    fast_budget: int | None = None,
) -> None:
    if not failed_items:
        return
    if fast_mode:
        budget = fast_budget if fast_budget is not None else fast_repair_budget(len(failed_items))
        failed_items = trim_failed_items_for_fast_budget(failed_items, budget)
        if not failed_items:
            return
    RUNTIME_METRICS.bump("retry.ass_surgical.attempts")
    RUNTIME_METRICS.bump("retry.ass_surgical.items", len(failed_items))
    total = len(failed_items)
    if total <= 6 and not fast_mode:
        for item in failed_items:
            repaired = translate_ass_slots_single(client, item, summary, tone_guide, target_lang, options)
            if repaired is not None:
                item["forced_restored"] = repaired
        return
    if total > 20:
        cprint(console, f"Conjunto grande de reparacion ASS ({total}); aplicando fallback quirurgico por bloques.", "yellow")
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
    allow_partial: bool = True,
    max_split_depth: int | None = None,
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
        allow_partial=allow_partial,
        max_split_depth=max_split_depth,
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
    fast_mode: bool = False,
    progress_tracker: GlobalProgressTracker | None = None,
) -> Tuple[str, int]:
    blocks, items = parse_srt_ir(text, limit)
    if not items:
        return text, 0

    translatable_items = [item for item in items if not item.get("fixed_final_text")]
    for item in items:
        if item.get("fixed_final_text"):
            item["final_text"] = item["fixed_final_text"]
    deduped_items, dedupe_groups = dedupe_srt_translation_items(translatable_items)
    if len(deduped_items) < len(translatable_items):
        saved = len(translatable_items) - len(deduped_items)
        cprint(
            console,
            f"Deduplicacion SRT: {len(translatable_items)} -> {len(deduped_items)} bloque(s) unicos (ahorrados {saved} items LLM).",
            "yellow",
        )
        RUNTIME_METRICS.bump("srt.dedupe.saved_items", saved)
        RUNTIME_METRICS.bump("srt.dedupe.groups", len(dedupe_groups))

    repair_reasons = {"empty_output", "label_prefix", "unchanged"} if fast_mode else set(SRT_REPAIR_REASONS)
    max_split_depth = FAST_SPLIT_MAX_DEPTH if fast_mode else None
    ass_batch_cap = effective_batch_cap(batch_size, one_shot)
    batches = build_adaptive_batches(deduped_items, ass_batch_cap, options, one_shot, console)
    active_batch_cap = ass_batch_cap
    history: List[str] = []
    progress_ctx = DummyProgress() if progress_tracker is not None else progress_bar(console)
    if progress_tracker is not None:
        progress_tracker.stage_start("translate", max(1, len(deduped_items)))
    with progress_ctx as progress:
        task_id = progress.add_task("Traduccion", total=len(deduped_items)) if progress_tracker is None else 0
        for batch in batches:
            work_batches = batched(batch, active_batch_cap) if len(batch) > active_batch_cap else [batch]
            for work_batch in work_batches:
                RUNTIME_METRICS.bump("translate.top_level_batches", 1)
                RUNTIME_METRICS.bump("translate.top_level_batch_items", len(work_batch))
                split_before = RUNTIME_METRICS.counters.get("translate.split.recursions", 0)
                by_item_id: dict[int, str | None] = {}
                uncached_items: List[dict] = []
                uncached_sources: List[str] = []
                for item in work_batch:
                    cache_key = translation_cache_key(
                        "srt_block",
                        target_lang,
                        item["protected_text"],
                        item.get("expected_line_count", 1),
                    )
                    cached = translation_cache_get(cache_key)
                    if cached is not None:
                        by_item_id[id(item)] = cached
                        RUNTIME_METRICS.bump("translate.cache.hits")
                        continue
                    uncached_items.append(item)
                    uncached_sources.append(item["protected_text"])
                    RUNTIME_METRICS.bump("translate.cache.misses")
                if uncached_sources:
                    context_hint = rolling_context_snippet(history, rolling_context)
                    out = translate_json_batch_with_split(
                        client,
                        uncached_sources,
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                        mode="srt_block",
                        rolling_context=context_hint,
                        allow_partial=True,
                        max_split_depth=max_split_depth,
                    )
                    for item, candidate in zip(uncached_items, out):
                        by_item_id[id(item)] = candidate
                for item in work_batch:
                    candidate = by_item_id.get(id(item))
                    status = scan_srt_candidate(item, candidate or "", target_lang)
                    item["candidate"] = status["candidate"]
                    item["status"] = status
                mark_duplicate_srt_batch_outputs(work_batch)
                for item in work_batch:
                    bump_status_reason_counters(item["status"])

                failed_items = [
                    item
                    for item in work_batch
                    if item_needs_llm_repair(item, repair_reasons, target_lang)
                ]
                split_after = RUNTIME_METRICS.counters.get("translate.split.recursions", 0)
                severe_ratio = len(failed_items) / float(max(1, len(work_batch)))
                if (
                    not fast_mode
                    and active_batch_cap > ADAPTIVE_BATCH_MIN_ITEMS
                    and (
                        severe_ratio >= SRT_ADAPTIVE_REDUCE_FAIL_RATE
                        or (
                            split_after > split_before
                            and severe_ratio >= SRT_ADAPTIVE_REDUCE_SPLIT_FAIL_RATE
                        )
                    )
                ):
                    new_cap = reduce_adaptive_batch_cap(active_batch_cap)
                    if new_cap < active_batch_cap:
                        cprint(
                            console,
                            (
                                f"Tope adaptativo de lote: {active_batch_cap} -> {new_cap} "
                                f"(split={split_after - split_before}, tasa_fallo={severe_ratio:.0%})"
                            ),
                            "yellow",
                        )
                        active_batch_cap = new_cap

                repair_limit = max(8, int(math.ceil(0.10 * max(1, len(work_batch)))))
                if fast_mode and failed_items:
                    failed_items = trim_failed_items_for_fast_budget(failed_items, fast_repair_budget(len(work_batch)))
                if failed_items and len(failed_items) > repair_limit:
                    cprint(
                        console,
                        f"Tasa alta de fallos: {len(failed_items)}/{len(work_batch)} SRT marcado(s). Reintentando lote @temp=0...",
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
                        allow_partial=True,
                        max_split_depth=max_split_depth,
                    )
                    failed_items = [
                        item
                        for item in work_batch
                        if item_needs_llm_repair(item, repair_reasons, target_lang)
                    ]
                    if fast_mode and failed_items:
                        failed_items = trim_failed_items_for_fast_budget(failed_items, fast_repair_budget(len(work_batch)))
                if failed_items and not fast_mode:
                    linewise_items = [
                        item
                        for item in failed_items
                        if needs_llm_repair_for_reasons(item["status"], SRT_LINEWISE_REPAIR_REASONS)
                    ]
                    for item in linewise_items:
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

                for item in work_batch:
                    if "forced_text" in item:
                        final_text = item["forced_text"]
                    elif item["status"]["ok"] or not item_needs_llm_repair(item, repair_reasons, target_lang):
                        final_text = enforce_line_count(
                            item["status"]["candidate"],
                            item["expected_line_count"],
                        )
                    else:
                        if fast_mode:
                            item["final_text"] = item["original_text_field"]
                            history.append(item["final_text"].replace("\n", " ").strip())
                            continue
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
                    if final_text and final_text != item["original_text_field"]:
                        cache_key = translation_cache_key(
                            "srt_block",
                            target_lang,
                            item["protected_text"],
                            item.get("expected_line_count", 1),
                        )
                        translation_cache_put(cache_key, final_text)
                        RUNTIME_METRICS.bump("translate.cache.writes")
                    history.append(final_text.replace("\n", " ").strip())
                if progress_tracker is not None:
                    progress_tracker.stage_advance(len(work_batch))
                else:
                    progress.advance(task_id, len(work_batch))
    if progress_tracker is not None:
        progress_tracker.stage_done()

    if dedupe_groups:
        for grouped_items in dedupe_groups.values():
            if len(grouped_items) <= 1:
                continue
            leader = grouped_items[0]
            leader_final = leader.get("final_text")
            for clone in grouped_items[1:]:
                if clone.get("final_text"):
                    continue
                clone["final_text"] = leader_final if leader_final is not None else clone["original_text_field"]

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
    fast_mode: bool = False,
    progress_tracker: GlobalProgressTracker | None = None,
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
    deduped_items, dedupe_groups = dedupe_ass_translation_items(translatable_items)
    if len(deduped_items) < len(translatable_items):
        saved = len(translatable_items) - len(deduped_items)
        cprint(
            console,
            f"Deduplicacion ASS: {len(translatable_items)} -> {len(deduped_items)} linea(s) unicas (ahorrados {saved} items LLM).",
            "yellow",
        )
        RUNTIME_METRICS.bump("ass.dedupe.saved_items", saved)
        RUNTIME_METRICS.bump("ass.dedupe.groups", len(dedupe_groups))

    ass_batch_cap = effective_batch_cap(batch_size, one_shot)
    batches = build_adaptive_batches(deduped_items, ass_batch_cap, options, one_shot, console)
    repair_reasons = (
        {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged", "short_line_overflow"}
        if fast_mode
        else {"empty_output", "placeholder_mismatch", "placeholder_artifacts", "unexpected_ass_markup", "unchanged", "language_leak", "very_suspicious", "short_line_overflow"}
    )
    max_split_depth = FAST_SPLIT_MAX_DEPTH if fast_mode else None
    active_batch_cap = ass_batch_cap
    history: List[str] = []

    progress_ctx = DummyProgress() if progress_tracker is not None else progress_bar(console)
    if progress_tracker is not None:
        progress_tracker.stage_start("translate", max(1, len(deduped_items)))
    with progress_ctx as progress:
        task_id = progress.add_task("Traduccion", total=len(deduped_items)) if progress_tracker is None else 0
        for batch in batches:
            work_batches = batched(batch, active_batch_cap) if len(batch) > active_batch_cap else [batch]
            for work_batch in work_batches:
                RUNTIME_METRICS.bump("translate.top_level_batches", 1)
                RUNTIME_METRICS.bump("translate.top_level_batch_items", len(work_batch))
                split_before = RUNTIME_METRICS.counters.get("translate.split.recursions", 0)
                by_item_id: dict[int, str | None] = {}
                uncached_items: List[dict] = []
                uncached_sources: List[str] = []
                for item in work_batch:
                    cache_key = translation_cache_key("ass_protected", target_lang, item["protected_text"])
                    cached = translation_cache_get(cache_key)
                    if cached is not None:
                        by_item_id[id(item)] = cached
                        RUNTIME_METRICS.bump("translate.cache.hits")
                        continue
                    uncached_items.append(item)
                    uncached_sources.append(item["protected_text"])
                    RUNTIME_METRICS.bump("translate.cache.misses")
                if uncached_sources:
                    context_hint = rolling_context_snippet(history, rolling_context)
                    out = translate_json_batch_with_split(
                        client,
                        uncached_sources,
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                        mode="ass_protected",
                        rolling_context=context_hint,
                        allow_partial=True,
                        max_split_depth=max_split_depth,
                    )
                    for item, candidate in zip(uncached_items, out):
                        by_item_id[id(item)] = candidate
                for item in work_batch:
                    candidate = by_item_id.get(id(item))
                    candidate_text = candidate if candidate is not None else item["protected_text"]
                    item["text_field"] = item["original_text_field"]
                    item["protected"] = item["protected_text"]
                    status = scan_ass_line_candidate(item, candidate_text, target_lang)
                    item["candidate"] = status["candidate"]
                    item["status"] = status
                    bump_status_reason_counters(status)

                failed_items = [
                    item
                    for item in work_batch
                    if item_needs_llm_repair(item, repair_reasons, target_lang)
                ]
                split_after = RUNTIME_METRICS.counters.get("translate.split.recursions", 0)
                severe_ratio = len(failed_items) / float(max(1, len(work_batch)))
                if (
                    not fast_mode
                    and active_batch_cap > ADAPTIVE_BATCH_MIN_ITEMS
                    and (
                        severe_ratio >= ASS_ADAPTIVE_REDUCE_FAIL_RATE
                        or (
                            split_after > split_before
                            and severe_ratio >= ASS_ADAPTIVE_REDUCE_SPLIT_FAIL_RATE
                        )
                    )
                ):
                    new_cap = reduce_adaptive_batch_cap(active_batch_cap)
                    if new_cap < active_batch_cap:
                        cprint(
                            console,
                            (
                                f"Tope adaptativo de lote: {active_batch_cap} -> {new_cap} "
                                f"(split={split_after - split_before}, tasa_fallo={severe_ratio:.0%})"
                            ),
                            "yellow",
                        )
                        active_batch_cap = new_cap

                repair_limit = max(8, int(math.ceil(0.10 * max(1, len(work_batch)))))
                if fast_mode and failed_items:
                    failed_items = trim_failed_items_for_fast_budget(failed_items, fast_repair_budget(len(work_batch)))
                if failed_items:
                    if len(failed_items) > repair_limit:
                        cprint(
                            console,
                            f"Tasa alta de fallos: {len(failed_items)}/{len(work_batch)} ASS marcado(s). Reintentando lote @temp=0...",
                            "yellow",
                        )
                    if len(failed_items) > repair_limit or not fast_mode:
                        retry_failed_items_batch(
                            client,
                            failed_items,
                            summary,
                            tone_guide,
                            target_lang,
                            options,
                            mode="ass_protected",
                            format_mode="schema",
                            allow_partial=True,
                            max_split_depth=max_split_depth,
                        )
                        failed_items = [
                            item
                            for item in work_batch
                            if item_needs_llm_repair(item, repair_reasons, target_lang)
                        ]
                        if fast_mode and failed_items:
                            failed_items = trim_failed_items_for_fast_budget(failed_items, fast_repair_budget(len(work_batch)))
                if failed_items:
                    run_ass_surgical_fallback(
                        client,
                        failed_items,
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                        console,
                        fast_mode=fast_mode,
                        fast_budget=(fast_repair_budget(len(work_batch)) if fast_mode else None),
                    )

                for item in work_batch:
                    status = item["status"]
                    if "forced_restored" in item:
                        forced = item["forced_restored"]
                        if ass_structure_preserved(item["original_text_field"], forced):
                            final_text = forced
                        else:
                            final_text = item["original_text_field"]
                    elif status["ok"] or not item_needs_llm_repair(item, repair_reasons, target_lang):
                        final_text = status["restored"]
                    else:
                        if fast_mode:
                            item["final_text"] = item["original_text_field"]
                            history.append(ass_plain_text(item["final_text"]).replace("\n", " ").strip())
                            continue
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
                    if final_text and final_text != item["original_text_field"]:
                        protected_final, _ = ass_protect_tags(final_text)
                        if ass_placeholders_match(item["protected_text"], protected_final):
                            cache_key = translation_cache_key("ass_protected", target_lang, item["protected_text"])
                            translation_cache_put(cache_key, protected_final)
                            RUNTIME_METRICS.bump("translate.cache.writes")
                    history.append(ass_plain_text(final_text).replace("\n", " ").strip())
                if progress_tracker is not None:
                    progress_tracker.stage_advance(len(work_batch))
                else:
                    progress.advance(task_id, len(work_batch))
    if progress_tracker is not None:
        progress_tracker.stage_done()

    if dedupe_groups:
        for grouped_items in dedupe_groups.values():
            if len(grouped_items) <= 1:
                continue
            leader = grouped_items[0]
            leader_status = leader.get("status")
            leader_candidate = leader.get("candidate")
            if leader_candidate is None and isinstance(leader_status, dict):
                leader_candidate = leader_status.get("candidate")
            for clone in grouped_items[1:]:
                if clone.get("final_text"):
                    continue
                if leader_candidate is None:
                    clone["final_text"] = clone["original_text_field"]
                    continue
                clone["text_field"] = clone["original_text_field"]
                clone["protected"] = clone["protected_text"]
                clone_status = scan_ass_line_candidate(clone, leader_candidate, target_lang)
                clone["candidate"] = clone_status["candidate"]
                clone["status"] = clone_status
                if clone_status["ok"] or not item_needs_llm_repair(clone, repair_reasons, target_lang):
                    clone["final_text"] = clone_status["restored"]
                else:
                    clone["final_text"] = clone["original_text_field"]

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
            task_id = progress.add_task("Traduccion", total=len(segments))
            for batch in batched(segments, batch_size):
                out = translate_batch(client, batch, summary, tone_guide, target_lang, options)
                if out is None or len(out) != len(batch):
                    out = []
                    for item in batch:
                        out.append(translate_one(client, item, summary, tone_guide, target_lang, options))
                translations.extend(out)
                progress.advance(task_id, len(batch))
        cprint(console, "Aplicando validaciones de seguridad ASS...", "cyan")

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
        cprint(console, "Demasiadas lineas marcadas; se omite reparacion LLM para mantener rendimiento", "yellow")

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
    effect_idx = None
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
            effect_idx = format_fields.index("Effect") if "Effect" in format_fields else None
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
        effect = fields[effect_idx] if effect_idx is not None else ""
        if not should_translate_ass_dialogue(text_field, effect=effect, for_summary=True):
            continue
        lines.append(normalize_inline_text(ass_plain_text(text_field)))
    return lines


def collect_plain_lines_srt(text: str) -> List[str]:
    lines = []
    block = []
    for line in text.splitlines():
        if line.strip() == "":
            if len(block) >= 3:
                text_block = "\n".join(block[2:])
                if not should_skip_srt_block_translation(text_block):
                    lines.extend(block[2:])
            block = []
            continue
        block.append(line)
    if len(block) >= 3:
        text_block = "\n".join(block[2:])
        if not should_skip_srt_block_translation(text_block):
            lines.extend(block[2:])
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


def detect_language_from_filename(path: Path) -> str | None:
    name = path.stem
    for lang, pattern in FILENAME_LANG_HINTS:
        if pattern.search(name):
            return lang
    return None


def display_language_label(lang: str) -> str:
    mapping = {
        "English": "Ingles",
        "Spanish": "Espanol",
        "Unknown": "Desconocido",
    }
    return mapping.get(lang, lang)


def bulk_dir() -> Path:
    """Return the directory where bulk subtitle files live.

    Historically this script expects to be run from the project root, with
    subtitles under ./SUBS_BULK. In practice users often run it while already
    inside SUBS_BULK, so we treat the current directory as the bulk directory
    when it looks like one.
    """
    cwd = Path.cwd()
    if cwd.name.lower() == BULK_DIR_NAME.lower():
        return cwd
    default = cwd / BULK_DIR_NAME
    if default.exists():
        return default
    # Fallback: if the current directory already contains subtitles, use it.
    if any(cwd.glob("*.ass")) or any(cwd.glob("*.srt")):
        return cwd
    return default


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


def _parse_index_selection(raw: str, max_index: int) -> List[int]:
    """Parse selections like: '1 2 5', '1,3-6', 'all'.

    Returns a list of 1-based indexes (deduped, sorted). Empty list means "no indexes".
    """
    text = (raw or "").strip().lower()
    if not text:
        return []
    if text in {"all", "*"}:
        return list(range(1, max_index + 1))

    # Tokenize by commas/whitespace.
    tokens = [t for t in re.split(r"[,\s]+", text) if t]
    out: set[int] = set()
    for tok in tokens:
        if re.fullmatch(r"\d+", tok):
            idx = int(tok)
            if 1 <= idx <= max_index:
                out.add(idx)
            continue
        m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", tok)
        if m:
            a = int(m.group(1))
            b = int(m.group(2))
            lo, hi = (a, b) if a <= b else (b, a)
            lo = max(lo, 1)
            hi = min(hi, max_index)
            for idx in range(lo, hi + 1):
                out.add(idx)
            continue
        # Non-index token: let caller treat it as a path/glob.
        return []
    return sorted(out)


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
        cprint(console, "No se encontraron modelos instalados via `ollama list`.", "yellow")
        return default_model or input("Ingresa el nombre del modelo: ").strip()

    cprint(console, "Elige modelo:", "bold cyan")
    for i, name in enumerate(ordered, 1):
        console.print(f"  {i}) {name}")
    console.print("  0) Escribir modelo personalizado")

    default_idx = None
    if default_model in ordered:
        default_idx = ordered.index(default_model) + 1

    prompt = "Modelo"
    while True:
        suffix = f" [predeterminado {default_idx}]" if default_idx else ""
        raw = input(f"{prompt}{suffix}: ").strip()
        if raw == "" and default_idx:
            return ordered[default_idx - 1]
        if raw.isdigit():
            num = int(raw)
            if num == 0:
                custom = input("Ingresa el nombre del modelo: ").strip()
                if custom:
                    return custom
            elif 1 <= num <= len(ordered):
                return ordered[num - 1]
        elif raw:
            return raw
        cprint(console, "Opcion invalida. Intenta de nuevo.", "yellow")


def prompt_choice(
    console,
    prompt: str,
    max_value: int,
    default_value: int | None = None,
) -> int:
    while True:
        suffix = f" [predeterminado {default_value}]" if default_value else ""
        choice = input(f"{prompt}{suffix}: ").strip()
        if choice == "" and default_value is not None:
            return default_value
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= max_value:
                return num
        cprint(console, "Opcion invalida. Intenta de nuevo.", "yellow")


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


def interactive_flow(args, console) -> Tuple[List[Path], Path | None, str, int | None, bool, str]:
    files = list_subtitle_files([".srt", ".ass"])
    in_paths: List[Path] = []
    if files:
        print_subtitle_options_table(console, files)
        ass_count = sum(1 for p in files if p.suffix.lower() == ".ass")
        srt_count = sum(1 for p in files if p.suffix.lower() == ".srt")
        show_metrics_cards(
            console,
            [
                ("Archivos", str(len(files)), None),
                ("ASS", str(ass_count), None),
                ("SRT", str(srt_count), None),
            ],
            title="Vista rapida",
            columns=3,
        )
        raw = input("Selecciona numero(s) de archivo (ej. 1 3 5, 1-4) o escribe una ruta (o 'all'): ").strip()

        idxs = _parse_index_selection(raw, len(files))
        if idxs:
            in_paths = [files[i - 1] for i in idxs]
        elif raw:
            candidate = resolve_input_path(raw)
            if candidate.exists():
                in_paths = [candidate]
            else:
                matches = [f for f in files if fnmatch.fnmatch(f.name, raw)]
                in_paths = matches
        else:
            if len(files) == 1:
                in_paths = [files[0]]
    if not in_paths:
        raw = input("Ingresa ruta de subtitulos (o 'all'): ").strip()
        idxs = _parse_index_selection(raw, len(files))
        if idxs and files:
            in_paths = [files[i - 1] for i in idxs]
        elif raw and raw.strip().lower() in {"all", "*"} and files:
            in_paths = files[:]
        else:
            in_paths = [resolve_input_path(raw)]

    missing = [p for p in in_paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Entrada no encontrada: {missing[0]}")

    sample_path = in_paths[0]
    text, _, _, _ = read_text(sample_path)
    ext = sample_path.suffix.lower()
    if ext == ".ass":
        sample_lines = collect_plain_lines_ass(text)
    else:
        sample_lines = collect_plain_lines_srt(text)

    preview_pool = sample_lines
    if len(sample_lines) > 80:
        sampled_preview, _ = prepare_summary_lines(sample_lines, 6000)
        if sampled_preview:
            preview_pool = sampled_preview

    cprint(console, "\nMuestra (primeras 8 lineas):", "bold cyan")
    for line in preview_pool[:8]:
        console.print(f"  {line}")

    sample_blob = "\n".join(preview_pool[:50])
    detected, confidence = detect_language(sample_blob)
    hint_lang = detect_language_from_filename(sample_path)
    if hint_lang and (detected == "Unknown" or confidence < 0.35):
        detected = hint_lang
        confidence = max(confidence, 0.55)
    conf_pct = int(confidence * 100)
    detected_label = display_language_label(detected)
    cprint(
        console,
        f"\nIdioma de entrada detectado: {detected_label} ({conf_pct}% confianza)",
        "bold green",
    )

    default_out = 1
    if detected == "Spanish":
        default_out = 2
    cprint(console, "Elige idioma de salida:", "bold cyan")
    console.print("  1) Espanol")
    console.print("  2) Ingles")
    out_choice = prompt_choice(console, "Idioma de salida", 2, default_out)
    target_lang = "Spanish" if out_choice == 1 else "English"

    cprint(console, "Traducir muestra o completo?", "bold cyan")
    console.print("  1) Muestra (10 lineas)")
    console.print("  2) Completo")
    scope_choice = prompt_choice(console, "Alcance", 2, 2)
    limit = 10 if scope_choice == 1 else None
    skip_summary = scope_choice == 1
    show_execution_roadmap(
        console,
        include_summary=(not skip_summary),
        multi_file=(len(in_paths) > 1),
    )

    model = choose_model(console, args.model)

    out_path = None
    if len(in_paths) == 1:
        out_path = resolve_output_path(args.out_path) if args.out_path else build_output_path(sample_path, target_lang)
        cprint(console, f"Archivo de salida: {out_path}", "bold cyan")
    else:
        cprint(console, f"Archivos seleccionados: {len(in_paths)}", "bold cyan")
        cprint(console, "Se generara un archivo de salida por cada entrada.", "bold cyan")
    if skip_summary:
        cprint(console, "Modo muestra: se omite resumen para mayor velocidad.", "yellow")
    return in_paths, out_path, target_lang, limit, skip_summary, model


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
        workers = max(1, int(getattr(args, "parallel_files", 1) or 1))
        if workers > 1:
            # Evita sobre-suscripcion de CPU al correr varios archivos en paralelo.
            args.num_threads = max(2, cpu_threads // workers)
        else:
            args.num_threads = cpu_threads
    if args.format_mode == "auto":
        args.format_mode = "schema"
    if args.rolling_context > 0:
        args.rolling_context = 0
    if not args.skip_summary:
        args.skip_summary = True
    if not args.one_shot:
        args.one_shot = True
    gpu_display = args.num_gpu if args.num_gpu is not None else "auto"
    cprint(
        console,
        f"Perfil rapido: batch={args.batch_size}, ctx={args.num_ctx}, predict={args.num_predict}, "
        f"hilos={args.num_threads}, gpu={gpu_display}, contexto={args.rolling_context}, resumen=off, one_shot=on",
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


def build_single_file_subprocess_cmd(args, in_path: Path, out_path: Path) -> List[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--in",
        str(in_path),
        "--out",
        str(out_path),
        "--model",
        str(args.model),
        "--host",
        str(args.host),
        "--target",
        str(args.target),
        "--batch-size",
        str(args.batch_size),
        "--summary-chars",
        str(args.summary_chars),
        "--timeout",
        str(args.timeout),
        "--keep-alive",
        str(args.keep_alive),
        "--temperature",
        str(args.temperature),
        "--ass-mode",
        str(args.ass_mode),
        "--rolling-context",
        str(args.rolling_context),
        "--format-mode",
        str(args.format_mode),
        "--parallel-files",
        "1",
    ]
    if args.num_predict is not None:
        cmd.extend(["--num-predict", str(args.num_predict)])
    if args.num_ctx is not None:
        cmd.extend(["--num-ctx", str(args.num_ctx)])
    if args.num_threads is not None:
        cmd.extend(["--num-threads", str(args.num_threads)])
    if args.num_gpu is not None:
        cmd.extend(["--num-gpu", str(args.num_gpu)])
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.skip_summary:
        cmd.append("--skip-summary")
    if args.fast:
        cmd.append("--fast")
    if args.one_shot:
        cmd.append("--one-shot")
    if args.minify_json:
        cmd.append("--minify-json")
    else:
        cmd.append("--no-minify-json")
    if args.bench:
        cmd.append("--bench")
    return cmd


def run_subprocess_translation(cmd: List[str]) -> Tuple[int, float, str]:
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    elapsed = time.perf_counter() - started
    combined_output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, elapsed, combined_output


def extract_translation_summary(output: str) -> str:
    if not output:
        return ""
    lines = output.splitlines()
    patterns = (
        "Tiempo (traduccion):",
        "Elapsed (translate):",
        "Contadores de reintentos:",
        "Retry counters:",
        "Presupuesto rapido de reintentos:",
        "Fast retry budget:",
        "Estadisticas de split:",
        "Split stats:",
        "Motivos mas marcados:",
        "Top flagged reasons:",
        "Llamadas a Ollama:",
        "Ollama calls:",
    )
    selected = [line.strip() for line in lines if any(p in line for p in patterns)]
    if not selected:
        return ""
    return " | ".join(selected)


def format_global_file_progress(done: int, total: int, ok: int, skipped: int, failed: int) -> str:
    if total <= 0:
        pct = 100.0
        remaining = 0
    else:
        pct = (done / float(total)) * 100.0
        remaining = max(0, total - done)
    return (
        f"Progreso global: {done}/{total} ({pct:.0f}%) | "
        f"ok={ok}, omitidos={skipped}, fallidos={failed}, restantes={remaining}"
    )


def print_global_file_progress(console, done: int, total: int, ok: int, skipped: int, failed: int) -> None:
    style = "bold cyan" if failed == 0 else "bold yellow"
    cprint(console, format_global_file_progress(done, total, ok, skipped, failed), style)


def translate_many_files_parallel_subprocess(
    console,
    args,
    jobs: List[Tuple[Path, Path]],
    selected_total: int,
    skipped: int = 0,
) -> Tuple[int, int, List[dict]]:
    max_workers = max(1, int(args.parallel_files or 1))
    total = len(jobs)
    cprint(
        console,
        f"Modo paralelo de archivos: workers={max_workers}, trabajos={total}, seleccionados={selected_total}",
        "bold cyan",
    )
    ok = 0
    failed = 0
    done = 0
    file_results: List[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        future_map = {
            pool.submit(run_subprocess_translation, build_single_file_subprocess_cmd(args, in_path, out_path)): (in_path, out_path)
            for in_path, out_path in jobs
        }
        for future in as_completed(future_map):
            done += 1
            in_path, _ = future_map[future]
            try:
                code, elapsed, output = future.result()
            except Exception as exc:
                failed += 1
                file_results.append(
                    {
                        "file_path": in_path,
                        "file_name": in_path.name,
                        "out_path": None,
                        "code": -1,
                        "translated_blocks": None,
                        "translate_elapsed": None,
                        "total_elapsed": 0.0,
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "cache_writes": 0,
                    }
                )
                cprint(console, f"[{done}/{total}] ERROR {in_path.name}: {exc}", "bold red")
                print_global_file_progress(
                    console,
                    skipped + done,
                    selected_total,
                    ok,
                    skipped,
                    failed,
                )
                continue
            if code == 0:
                ok += 1
                summary = extract_translation_summary(output)
                file_results.append(
                    {
                        "file_path": in_path,
                        "file_name": in_path.name,
                        "out_path": None,
                        "code": 0,
                        "translated_blocks": None,
                        "translate_elapsed": None,
                        "total_elapsed": float(elapsed),
                        "cache_hits": 0,
                        "cache_misses": 0,
                        "cache_writes": 0,
                    }
                )
                cprint(console, f"[{done}/{total}] OK {in_path.name} ({elapsed:.1f}s)", "green")
                if summary:
                    console.print(summary)
                print_global_file_progress(
                    console,
                    skipped + done,
                    selected_total,
                    ok,
                    skipped,
                    failed,
                )
                continue
            failed += 1
            file_results.append(
                {
                    "file_path": in_path,
                    "file_name": in_path.name,
                    "out_path": None,
                    "code": int(code),
                    "translated_blocks": None,
                    "translate_elapsed": None,
                    "total_elapsed": float(elapsed),
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "cache_writes": 0,
                }
            )
            cprint(console, f"[{done}/{total}] ERROR {in_path.name} (exit={code}, {elapsed:.1f}s)", "bold red")
            if output.strip():
                tail = "\n".join(output.splitlines()[-30:])
                console.print(tail)
            print_global_file_progress(
                console,
                skipped + done,
                selected_total,
                ok,
                skipped,
                failed,
            )
    return ok, failed, file_results


def translate_single_file(
    client,
    console,
    args,
    in_path: Path,
    out_path: Path,
    progress_tracker: GlobalProgressTracker | None = None,
) -> Tuple[int, dict]:
    RUNTIME_METRICS.reset()
    text, line_ending, final_newline, bom = read_text(in_path)
    ext = in_path.suffix.lower()
    options = build_ollama_options(args)
    start_total = time.perf_counter()
    file_stats: dict = {
        "file_path": in_path,
        "file_name": in_path.name,
        "out_path": out_path,
        "code": 1,
        "translated_blocks": 0,
        "translate_elapsed": 0.0,
        "total_elapsed": 0.0,
        "cache_hits": 0,
        "cache_misses": 0,
        "cache_writes": 0,
    }

    summary = ""
    tone_guide = ""
    srt_rolling_context = args.rolling_context
    if ext == ".srt" and srt_rolling_context > 0:
        if progress_tracker is None:
            cprint(
                console,
                "Modo SRT: se fuerza rolling-context=0 para evitar deriva entre bloques.",
                "yellow",
            )
        srt_rolling_context = 0
    if not args.skip_summary:
        if progress_tracker is None:
            cprint(console, "Construyendo resumen...", "bold cyan")
        with RUNTIME_METRICS.timed("stage.summary"):
            if ext == ".ass":
                plain_lines = collect_plain_lines_ass(text)
            elif ext == ".srt":
                plain_lines = collect_plain_lines_srt(text)
            else:
                print("Tipo de archivo no soportado. Usa .ass o .srt", file=sys.stderr)
                file_stats["code"] = 2
                file_stats["total_elapsed"] = time.perf_counter() - start_total
                return 2, file_stats
            summary = summarize_subs(
                client,
                plain_lines,
                args.summary_chars,
                options,
                console,
                progress_tracker=progress_tracker,
            )
        if progress_tracker is None:
            cprint(console, "Resumen listo.", "bold green")
        with RUNTIME_METRICS.timed("stage.tone_guide"):
            tone_guide = build_tone_guide(
                client,
                summary,
                plain_lines,
                options,
                console,
                progress_tracker=progress_tracker,
            )
        if tone_guide and progress_tracker is None:
            cprint(console, "Guia de tono lista.", "bold green")

    if progress_tracker is None:
        cprint(console, "Traduciendo...", "bold cyan")
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
                args.fast,
                progress_tracker=progress_tracker,
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
                srt_rolling_context,
                args.fast,
                progress_tracker=progress_tracker,
            )
        else:
            print("Tipo de archivo no soportado. Usa .ass o .srt", file=sys.stderr)
            file_stats["code"] = 2
            file_stats["total_elapsed"] = time.perf_counter() - start_total
            return 2, file_stats

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(out_path, out_text.splitlines(), line_ending, final_newline, bom)
    translate_elapsed = RUNTIME_METRICS.seconds.get("stage.translate", 0.0)
    total_elapsed = time.perf_counter() - start_total
    cprint(console, f"Bloques traducidos: {translated_count}", "bold green")
    cprint(console, f"Archivo generado: {out_path}", "bold green")
    cprint(console, f"Tiempo (traduccion): {translate_elapsed:.1f}s", "bold green")
    cprint(console, f"Tiempo (total): {total_elapsed:.1f}s", "bold green")
    print_runtime_breakdown(console, total_elapsed)
    file_stats["code"] = 0
    file_stats["translated_blocks"] = int(translated_count)
    file_stats["translate_elapsed"] = float(translate_elapsed)
    file_stats["total_elapsed"] = float(total_elapsed)
    file_stats["cache_hits"] = int(RUNTIME_METRICS.counters.get("translate.cache.hits", 0))
    file_stats["cache_misses"] = int(RUNTIME_METRICS.counters.get("translate.cache.misses", 0))
    file_stats["cache_writes"] = int(RUNTIME_METRICS.counters.get("translate.cache.writes", 0))
    return 0, file_stats


def run_multi_file_jobs(
    client,
    console,
    args,
    jobs: List[Tuple[Path, Path]],
    selected_total: int,
    skipped: int,
) -> Tuple[int, int, List[dict]]:
    ok = 0
    failed = 0
    file_results: List[dict] = []
    if args.parallel_files > 1 and len(jobs) > 1:
        if selected_total > 1:
            print_global_file_progress(console, skipped, selected_total, ok, skipped, failed)
        return translate_many_files_parallel_subprocess(
            console,
            args,
            jobs,
            selected_total=selected_total,
            skipped=skipped,
        )

    use_persistent = RICH_AVAILABLE and selected_total > 1
    if use_persistent:
        with GlobalProgressTracker(console, selected_total, skipped=skipped) as tracker:
            tracker.set_counts(ok, skipped, failed)
            for file_path, out_file in jobs:
                next_global = skipped + ok + failed + 1
                tracker.start_file(
                    file_path,
                    next_global,
                    selected_total,
                    include_summary=(not args.skip_summary),
                )
                code, stats = translate_single_file(
                    client,
                    console,
                    args,
                    file_path,
                    out_file,
                    progress_tracker=tracker,
                )
                file_results.append(stats)
                if code == 0:
                    ok += 1
                else:
                    failed += 1
                tracker.set_counts(ok, skipped, failed)
        return ok, failed, file_results

    if selected_total > 1:
        print_global_file_progress(console, skipped, selected_total, ok, skipped, failed)
    for file_path, out_file in jobs:
        if selected_total > 1:
            next_global = skipped + ok + failed + 1
            cprint(console, f"\n=== Traduciendo [{next_global}/{selected_total}]: {file_path.name} ===", "bold cyan")
        else:
            cprint(console, f"\n=== Traduciendo: {file_path.name} ===", "bold cyan")
        code, stats = translate_single_file(client, console, args, file_path, out_file)
        file_results.append(stats)
        if code == 0:
            ok += 1
        else:
            failed += 1
        if selected_total > 1:
            print_global_file_progress(
                console,
                skipped + ok + failed,
                selected_total,
                ok,
                skipped,
                failed,
            )
    return ok, failed, file_results


def main() -> int:
    console = get_console()
    RUNTIME_METRICS.reset()
    parser = argparse.ArgumentParser(description="Traduce subtitulos .ASS/.SRT usando Ollama local.")
    parser.add_argument("--in", dest="in_path", help="Archivo de entrada (o patron glob con --batch)")
    parser.add_argument("--out", dest="out_path", help="Archivo de salida")
    parser.add_argument("--model", default="gemma3:4b", help="Nombre del modelo en Ollama")
    parser.add_argument("--host", default="http://localhost:11434", help="Host de Ollama")
    parser.add_argument("--target", default="Spanish", help="Idioma objetivo")
    parser.add_argument("--batch-size", type=int, default=256, help="Tope de tamano de lote para traduccion")
    parser.add_argument("--summary-chars", type=int, default=6000, help="Maximo de caracteres por chunk de resumen")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout HTTP en segundos")
    parser.add_argument("--keep-alive", default="10m", help="Valor keep_alive de Ollama (ej. 10m, 0)")
    parser.add_argument("--temperature", type=float, default=0.1, help="Temperatura del LLM")
    parser.add_argument("--num-predict", type=int, help="Limite de tokens generados por respuesta")
    parser.add_argument("--num-ctx", type=int, help="Ventana de contexto")
    parser.add_argument("--num-threads", type=int, default=6, help="Hilos para ejecucion del modelo")
    parser.add_argument("--num-gpu", type=int, help="Capas GPU para ejecucion del modelo")
    parser.add_argument("--limit", type=int, help="Traduce solo los primeros N bloques de dialogo")
    parser.add_argument("--skip-summary", action="store_true", help="Omite el paso de resumen")
    parser.add_argument("--interactive", action="store_true", help="Modo interactivo")
    parser.add_argument("--batch", action="store_true", help="Traduce todos los subtitulos de SUBS_BULK")
    parser.add_argument(
        "--parallel-files",
        type=int,
        default=3,
        help="Traduce multiples archivos en paralelo para --batch o modo multiarchivo (workers por subprocess)",
    )
    parser.add_argument("--overwrite", action="store_true", help="Sobrescribe salidas en modo --batch")
    parser.add_argument("--ass-mode", choices=["line", "segment"], default="line", help="Modo de traduccion ASS")
    fast_group = parser.add_mutually_exclusive_group()
    fast_group.add_argument(
        "--fast",
        dest="fast",
        action="store_true",
        default=True,
        help="Aplica valores predeterminados de perfil rapido (predeterminado: on)",
    )
    fast_group.add_argument(
        "--no-fast",
        dest="fast",
        action="store_false",
        help="Desactiva el perfil rapido predeterminado",
    )
    parser.add_argument("--one-shot", action="store_true", help="Fuerza un solo lote cuando entra en contexto")
    parser.add_argument("--rolling-context", type=int, default=0, help="Usa las ultimas N lineas traducidas como contexto")
    parser.add_argument(
        "--format-mode",
        choices=["auto", "json", "schema"],
        default="auto",
        help="Modo de salida estructurada: auto=json rapido + reintento schema, json=siempre json, schema=siempre schema",
    )
    parser.add_argument(
        "--minify-json",
        dest="minify_json",
        action="store_true",
        default=True,
        help="Minifica el JSON embebido en prompts (predeterminado: on)",
    )
    parser.add_argument(
        "--no-minify-json",
        dest="minify_json",
        action="store_false",
        help="Deshabilita JSON minificado en prompts",
    )
    parser.add_argument("--bench", action="store_true", help="Activa logs detallados de bench por llamada")
    parser.add_argument("--self-test", action="store_true", help="Ejecuta auto-pruebas internas y sale")
    args = parser.parse_args()
    set_runtime_flags(args.format_mode, args.minify_json, args.bench)

    if args.self_test:
        self_test_ass_repair_snippet()
        self_test_hybrid_pipeline()
        cprint(console, "Auto-prueba OK", "bold green")
        return 0

    if args.batch and args.interactive:
        print("No se puede combinar --batch con --interactive", file=sys.stderr)
        return 2
    if args.batch and args.out_path:
        print("--out no esta soportado en modo --batch", file=sys.stderr)
        return 2
    if args.parallel_files < 1:
        print("--parallel-files debe ser >= 1", file=sys.stderr)
        return 2

    if args.interactive or not args.in_path:
        show_app_header(console)

    in_paths: List[Path] = []
    out_path = None
    if not args.batch:
        if args.interactive or not args.in_path:
            try:
                in_paths, out_path, args.target, args.limit, skip_summary, model = interactive_flow(
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
                print(f"Entrada no encontrada: {in_path}", file=sys.stderr)
                return 2
            in_paths = [in_path]
            out_path = resolve_output_path(args.out_path) if args.out_path else build_output_path(in_path, args.target)

    apply_fast_profile(args, console)
    set_runtime_flags(args.format_mode, args.minify_json, args.bench)
    if args.bench:
        cprint(
            console,
            (
                f"Modo bench ON | format_mode={resolve_format_mode()} | "
                f"minify_json={'on' if MINIFY_JSON_PROMPTS else 'off'}"
            ),
            "bold cyan",
        )

    client = OllamaClient(args.host, args.model, args.timeout, args.keep_alive)
    if args.num_ctx is not None:
        set_runtime_default_ctx_tokens(args.num_ctx)
    else:
        detected_ctx = client.detect_model_num_ctx()
        if detected_ctx is not None:
            set_runtime_default_ctx_tokens(detected_ctx)
            cprint(console, f"Contexto del modelo detectado automaticamente: num_ctx={detected_ctx}", "cyan")
        else:
            set_runtime_default_ctx_tokens(DEFAULT_CTX_TOKENS)

    if args.batch:
        files = collect_batch_inputs(args.in_path, args.target)
        if not files:
            cprint(console, "No se encontraron subtitulos para traduccion por lotes.", "yellow")
            return 0

        selected_total = len(files)
        skipped = 0
        jobs: List[Tuple[Path, Path]] = []
        for file_path in files:
            out_file = build_output_path(file_path, args.target)
            if out_file.exists() and not args.overwrite:
                skipped += 1
                cprint(console, f"Omitido (ya existe): {out_file.name}", "yellow")
                continue
            jobs.append((file_path, out_file))

        show_metrics_cards(
            console,
            [
                ("Seleccionados", str(selected_total), None),
                ("A procesar", str(len(jobs)), None),
                ("Omitidos", str(skipped), "salida ya existente"),
            ],
            title="Resumen previo del lote",
            columns=3,
        )
        show_execution_roadmap(
            console,
            include_summary=(not args.skip_summary),
            multi_file=True,
        )

        ok, failed, file_results = run_multi_file_jobs(
            client,
            console,
            args,
            jobs,
            selected_total=selected_total,
            skipped=skipped,
        )

        print_multi_file_final_summary(
            console,
            title="Resumen final del lote",
            selected_total=selected_total,
            to_process=len(jobs),
            skipped=skipped,
            ok=ok,
            failed=failed,
            file_results=file_results,
        )
        return 0 if failed == 0 else 1

    if not in_paths:
        print("No se selecciono ninguna entrada.", file=sys.stderr)
        return 2

    # If the user selected multiple files interactively, run a mini-batch.
    if len(in_paths) > 1:
        if args.out_path:
            print("--out no esta soportado al traducir multiples entradas", file=sys.stderr)
            return 2

        selected_total = len(in_paths)
        skipped = 0
        jobs: List[Tuple[Path, Path]] = []
        for file_path in in_paths:
            out_file = build_output_path(file_path, args.target)
            if out_file.exists() and not args.overwrite:
                skipped += 1
                cprint(console, f"Omitido (ya existe): {out_file.name}", "yellow")
                continue
            jobs.append((file_path, out_file))

        show_metrics_cards(
            console,
            [
                ("Seleccionados", str(selected_total), None),
                ("A procesar", str(len(jobs)), None),
                ("Omitidos", str(skipped), "salida ya existente"),
            ],
            title="Resumen previo multiarchivo",
            columns=3,
        )
        show_execution_roadmap(
            console,
            include_summary=(not args.skip_summary),
            multi_file=True,
        )

        ok, failed, file_results = run_multi_file_jobs(
            client,
            console,
            args,
            jobs,
            selected_total=selected_total,
            skipped=skipped,
        )

        print_multi_file_final_summary(
            console,
            title="Resumen final multiarchivo",
            selected_total=selected_total,
            to_process=len(jobs),
            skipped=skipped,
            ok=ok,
            failed=failed,
            file_results=file_results,
        )
        return 0 if failed == 0 else 1

    in_path = in_paths[0]
    if out_path is None:
        out_path = build_output_path(in_path, args.target)
    show_metrics_cards(
        console,
        [
            ("Entrada", in_path.name, None),
            ("Salida", out_path.name, None),
            ("Modo", "Muestra" if args.limit is not None else "Completo", None),
        ],
        title="Ejecucion de archivo unico",
        columns=3,
    )
    show_execution_roadmap(
        console,
        include_summary=(not args.skip_summary),
        multi_file=False,
    )
    code, _stats = translate_single_file(client, console, args, in_path, out_path)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
