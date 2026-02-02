#!/usr/bin/env python3
"""Translate .ASS or .SRT subtitles using a local Ollama model.

Workflow:
1) Read subtitles and build a brief summary for context.
2) Translate lines (SRT) or dialogue text segments (ASS), preserving tags.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple
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
ASS_PLACEHOLDER_TOKEN_RE = re.compile(r"__ASS_TAG_\d+__")
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
    def __init__(self, host: str, model: str, timeout: int) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    def chat(self, messages, options=None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if options:
            payload["options"] = options
        url = f"{self.host}/api/chat"
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode("utf-8", errors="replace")
        except error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Ollama HTTP {exc.code}: {body}") from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"Cannot reach Ollama at {self.host}. Is it running?"
            ) from exc
        try:
            payload = json.loads(body)
            return payload["message"]["content"]
        except Exception as exc:
            raise RuntimeError(f"Unexpected Ollama response: {body[:200]}") from exc


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
    chunks = build_chunks(lines, max_chars)
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
    expected = ASS_PLACEHOLDER_TOKEN_RE.findall(source_protected)
    actual = ASS_PLACEHOLDER_TOKEN_RE.findall(translated)
    return expected == actual


def has_ass_markup(text: str) -> bool:
    return ASS_TAG_RE.search(text) is not None


def ass_structure_preserved(source: str, translated: str) -> bool:
    return ASS_TAG_RE.findall(source) == ASS_TAG_RE.findall(translated)


def has_placeholder_artifacts(text: str) -> bool:
    return "ASS_TAG" in text


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
            candidate = translate_one(client, text, sum_ctx, tone_ctx, target_lang, options)
            SINGLE_TRANSLATION_CACHE[cache_key] = candidate
        if not candidate:
            continue
        if has_placeholder_artifacts(candidate) and "__ASS_TAG_" not in text:
            continue
        if not ass_placeholders_match(text, candidate):
            continue
        if validator and not validator(candidate):
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
            if (
                not has_ass_markup(candidate)
                and not has_placeholder_artifacts(candidate)
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
            validator=lambda cand: not has_ass_markup(cand) and not has_placeholder_artifacts(cand),
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
    has_placeholders = any("__ASS_TAG_" in item for item in texts)
    placeholder_rule = (
        "If placeholders like __ASS_TAG_0__ appear, keep them exactly. "
        if has_placeholders
        else "Do not introduce placeholders like __ASS_TAG_0__ if they are not in the source. "
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
        "Translate only the current line, never summarize a scene or multiple lines. "
        "Fully translate every word from the source; do not drop connectors or leave source fragments. "
        f"{placeholder_rule}"
        "Respect sarcasm, double meanings, and register (formal/informal) as guided; do not neutralize tone. "
        "Return ONLY a JSON array of strings, same length and order.\n\n"
        f"Input JSON: {json.dumps(texts, ensure_ascii=False)}"
    )
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
) -> str:
    summary_short = summary.strip()
    if len(summary_short) > 1200:
        summary_short = summary_short[:1200] + "..."
    tone_short = tone_guide.strip()
    if len(tone_short) > 800:
        tone_short = tone_short[:800] + "..."
    has_placeholders = "__ASS_TAG_" in text
    placeholder_rule = (
        "If placeholders like __ASS_TAG_0__ appear, keep them exactly.\n\n"
        if has_placeholders
        else "Do not introduce placeholders like __ASS_TAG_0__ if they are not in the source.\n\n"
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
        "Translate only this line, never summarize the plot or add narration. "
        "Fully translate every word from the source; do not drop connectors or leave source fragments. "
        f"{placeholder_rule}"
        "Respect sarcasm, double meanings, and register (formal/informal) as guided; do not neutralize tone.\n\n"
        f"Text: {text}"
    )
    return client.chat(
        [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ],
        options=options,
    ).strip()


def batched(items: List[str], size: int) -> List[List[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def should_translate(segment: str) -> bool:
    return bool(re.search(r"[A-Za-z]", segment))


def split_ass_text(text: str) -> List[str]:
    parts = ASS_TAG_RE.split(text)
    return [p for p in parts if p is not None and p != ""]


def ass_protect_tags(text: str) -> Tuple[str, List[Tuple[str, str]]]:
    replacements: List[Tuple[str, str]] = []

    def repl(match: re.Match) -> str:
        key = f"__ASS_TAG_{len(replacements)}__"
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
) -> Tuple[str, int]:
    lines = text.splitlines()
    blocks = []
    cur = []
    for line in lines:
        if line.strip() == "":
            if cur:
                blocks.append(cur)
                cur = []
        else:
            cur.append(line)
    if cur:
        blocks.append(cur)

    trans_targets = []
    trans_map = []
    count = 0
    for idx, block in enumerate(blocks):
        if len(block) < 2:
            continue
        if limit is not None and count >= limit:
            continue
        count += 1
        text_lines = block[2:] if len(block) > 2 else [""]
        joined = "\n".join(text_lines)
        trans_map.append((idx, joined))
        trans_targets.append(joined)

    translations = []
    if trans_targets:
        with progress_bar(console) as progress:
            task_id = progress.add_task("Translation", total=len(trans_targets))
            for batch in batched(trans_targets, batch_size):
                out = translate_batch(client, batch, summary, tone_guide, target_lang, options)
                if out is None or len(out) != len(batch):
                    out = []
                    for item in batch:
                        out.append(translate_one(client, item, summary, tone_guide, target_lang, options))
                translations.extend(out)
                progress.advance(task_id, len(batch))
        cprint(console, "Applying quality checks...", "cyan")

    for (block_idx, source_text), translated in zip(trans_map, translations):
        fixed = translated
        needs_retry = (
            is_suspicious_translation(source_text, translated)
            or not ass_placeholders_match(source_text, translated)
        )
        if needs_retry:
            retry = guarded_translate_one(
                client,
                source_text,
                summary,
                tone_guide,
                target_lang,
                options,
                source_for_checks=source_text,
                prefer_no_context=True,
            )
            fixed = retry if retry is not None else source_text
        block = blocks[block_idx]
        new_lines = fixed.split("\n") if fixed else [""]
        blocks[block_idx] = [block[0], block[1], *new_lines]

    out_lines = []
    for i, block in enumerate(blocks):
        out_lines.extend(block)
        if i != len(blocks) - 1:
            out_lines.append("")

    return "\n".join(out_lines), count


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

    lines = text.splitlines()
    in_events = False
    format_fields = None
    text_idx = None

    trans_targets = []
    trans_map = []
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

        text_field = fields[text_idx]
        if not should_translate(text_field):
            continue
        protected, replacements = ass_protect_tags(text_field)
        line_state[i] = {
            "kind": kind,
            "spaces": spaces,
            "fields": fields,
            "text_field": text_field,
            "protected": protected,
            "replacements": replacements,
            "text_idx": text_idx,
        }
        trans_map.append(i)
        trans_targets.append(protected)
        translated_count += 1

    translations = []
    if trans_targets:
        with progress_bar(console) as progress:
            task_id = progress.add_task("Translation", total=len(trans_targets))
            for batch in batched(trans_targets, batch_size):
                out = translate_batch(client, batch, summary, tone_guide, target_lang, options)
                if out is None or len(out) != len(batch):
                    out = []
                    for item in batch:
                        out.append(translate_one(client, item, summary, tone_guide, target_lang, options))
                translations.extend(out)
                progress.advance(task_id, len(batch))
        cprint(console, "Applying ASS safety checks...", "cyan")

    for line_idx, translated in zip(trans_map, translations):
        state = line_state.get(line_idx)
        if not state:
            continue
        placeholder_ok = ass_placeholders_match(state["protected"], translated)
        protected_retry = None
        did_retry_protected = False
        if not placeholder_ok or has_ass_markup(translated):
            protected_retry = guarded_translate_one(
                client,
                state["protected"],
                summary,
                tone_guide,
                target_lang,
                options,
                source_for_checks=ass_plain_text(state["text_field"]),
                validator=lambda cand, src=state["protected"]: ass_placeholders_match(src, cand)
                and not has_ass_markup(cand),
                prefer_no_context=True,
            )
            did_retry_protected = True
            if protected_retry is not None:
                translated = protected_retry
                placeholder_ok = True

        if placeholder_ok:
            restored = ass_restore_tags(translated, state["replacements"])
            source_plain = ass_plain_text(state["text_field"])
            restored_plain = ass_plain_text(restored)
            structure_ok = ass_structure_preserved(state["text_field"], restored)
            if (
                not structure_ok
                or has_placeholder_artifacts(restored)
                or is_suspicious_translation(source_plain, restored_plain)
            ):
                retry = None
                if not did_retry_protected:
                    retry = guarded_translate_one(
                        client,
                        state["protected"],
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                        source_for_checks=source_plain,
                        validator=lambda cand, src=state["protected"]: ass_placeholders_match(src, cand)
                        and not has_ass_markup(cand),
                        prefer_no_context=True,
                    )
                if retry is not None:
                    restored = ass_restore_tags(retry, state["replacements"])
                    if (
                        not ass_structure_preserved(state["text_field"], restored)
                        or has_placeholder_artifacts(restored)
                        or is_suspicious_translation(source_plain, ass_plain_text(restored))
                    ):
                        restored = translate_ass_text_field_segmented(
                            client,
                            state["text_field"],
                            summary,
                            tone_guide,
                            target_lang,
                            options,
                        )
                else:
                    restored = translate_ass_text_field_segmented(
                        client,
                        state["text_field"],
                        summary,
                        tone_guide,
                        target_lang,
                        options,
                    )
        else:
            restored = translate_ass_text_field_segmented(
                client,
                state["text_field"],
                summary,
                tone_guide,
                target_lang,
                options,
            )
        fields = state["fields"]
        fields[state["text_idx"]] = restored
        lines[line_idx] = f"{state['kind']}:{state['spaces']}{','.join(fields)}"

    return "\n".join(lines), translated_count


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
        if state:
            fixed = translated
            needs_retry = (
                is_suspicious_translation(source_token, translated)
                or not ass_placeholders_match(source_token, translated)
                or has_ass_markup(translated)
            )
            if needs_retry:
                retry = guarded_translate_one(
                    client,
                    source_token,
                    summary,
                    tone_guide,
                    target_lang,
                    options,
                    source_for_checks=source_token,
                    validator=lambda cand: not has_ass_markup(cand) and not has_placeholder_artifacts(cand),
                )
                fixed = retry if retry is not None else source_token
            state["tokens"][token_idx] = fixed

    for line_idx, state in line_state.items():
        new_text = "".join(state["tokens"])
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


def build_output_path(in_path: Path, target_lang: str) -> Path:
    lang = target_lang.strip().lower()
    suffix = in_path.suffix
    stem = in_path.stem
    out_dir = bulk_dir()
    if lang in ("spanish", "es", "es-419", "es_419"):
        return out_dir / f"{stem}_es-419{suffix}"
    if lang in ("english", "en", "en-us", "en_us", "en-gb", "en_gb"):
        return out_dir / f"{stem}_en{suffix}"
    return out_dir / f"{stem}.translated{suffix}"


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
    if args.num_gpu is None:
        args.num_gpu = 0
    cprint(
        console,
        f"Fast profile: batch={args.batch_size}, ctx={args.num_ctx}, predict={args.num_predict}, "
        f"threads={args.num_threads}, gpu={args.num_gpu}",
        "bold cyan",
    )


def main() -> int:
    console = get_console()
    parser = argparse.ArgumentParser(description="Translate .ASS/.SRT subtitles using local Ollama.")
    parser.add_argument("--in", dest="in_path", help="Input subtitle file")
    parser.add_argument("--out", dest="out_path", help="Output subtitle file")
    parser.add_argument("--model", default="gemma3:4b", help="Ollama model name")
    parser.add_argument("--host", default="http://localhost:11434", help="Ollama host")
    parser.add_argument("--target", default="Spanish", help="Target language")
    parser.add_argument("--batch-size", type=int, default=6, help="Batch size for translation")
    parser.add_argument("--summary-chars", type=int, default=6000, help="Max chars per summary chunk")
    parser.add_argument("--timeout", type=int, default=300, help="HTTP timeout seconds")
    parser.add_argument("--temperature", type=float, default=0.1, help="LLM temperature")
    parser.add_argument("--num-predict", type=int, help="Limit tokens generated per response")
    parser.add_argument("--num-ctx", type=int, help="Context window size")
    parser.add_argument("--num-threads", type=int, help="Threads for model execution")
    parser.add_argument("--num-gpu", type=int, help="GPU layers for model execution")
    parser.add_argument("--limit", type=int, help="Translate only the first N dialogue blocks")
    parser.add_argument("--skip-summary", action="store_true", help="Skip the summary step")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--ass-mode", choices=["line", "segment"], default="line", help="ASS translation mode")
    parser.add_argument("--fast", action="store_true", help="Apply fast profile defaults")
    args = parser.parse_args()

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

    text, line_ending, final_newline, bom = read_text(in_path)
    ext = in_path.suffix.lower()

    options = {"temperature": args.temperature}
    if args.num_predict is not None:
        options["num_predict"] = args.num_predict
    if args.num_ctx is not None:
        options["num_ctx"] = args.num_ctx
    if args.num_threads is not None:
        options["num_thread"] = args.num_threads
    if args.num_gpu is not None:
        options["num_gpu"] = args.num_gpu
    client = OllamaClient(args.host, args.model, args.timeout)

    # Build summary first
    summary = ""
    tone_guide = ""
    if not args.skip_summary:
        cprint(console, "Building summary...", "bold cyan")
        if ext == ".ass":
            plain_lines = collect_plain_lines_ass(text)
        elif ext == ".srt":
            plain_lines = collect_plain_lines_srt(text)
        else:
            print("Unsupported file type. Use .ass or .srt", file=sys.stderr)
            return 2
        summary = summarize_subs(client, plain_lines, args.summary_chars, options, console)
        cprint(console, "Summary ready.", "bold green")
        tone_guide = build_tone_guide(client, summary, plain_lines, options, console)
        if tone_guide:
            cprint(console, "Tone guide ready.", "bold green")

    cprint(console, "Translating...", "bold cyan")
    start = time.time()
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
        )
    else:
        print("Unsupported file type. Use .ass or .srt", file=sys.stderr)
        return 2

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(out_path, out_text.splitlines(), line_ending, final_newline, bom)
    elapsed = time.time() - start
    cprint(console, f"Translated blocks: {translated_count}", "bold green")
    cprint(console, f"Output written to: {out_path}", "bold green")
    cprint(console, f"Elapsed: {elapsed:.1f}s", "bold green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
