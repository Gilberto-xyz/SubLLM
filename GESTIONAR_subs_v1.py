#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GESTIONAR_subs_v1.py
Version 3.4 - 20-feb-2026

- Usa SUBS_BULK como carpeta de trabajo para videos y subtitulos.
- Extrae subtitulos de los videos detectados.
- Muxea subtitulos traducidos, fija metadatos y marca espanol como default.
- Puede reemplazar el video original (con backup opcional).
- Al final pregunta si deseas borrar subtitulos ya muxeados.

Mejoras 3.1:
- Escaneo mas rapido de pistas de subtitulos (cache + paralelo).
- En el listado de idiomas muestra cantidad de pistas por idioma.
- Tras extraer, muestra un resumen por video/idioma con peso y cantidad de lineas/entradas.

Mejoras 3.2:
- Seleccion automatica de la mejor pista/archivo cuando hay duplicados del mismo idioma.
  (Evita subs tipo SDH/full con canciones/SFX cuando existe una alternativa mas limpia.)

Mejoras 3.3:
- Escaneo de MKV via mkvmerge (rapido) cuando MKVToolNix esta disponible.
- Extraccion por video en lote (una invocacion por archivo) via mkvextract/ffmpeg.
- Omitir automaticamente subtitulos basados en imagen (PGS/VobSub/DVB) que requieren OCR.

Mejoras 3.4:
- Resumen de cobertura tras escaneo: pistas por video e idiomas compartidos/no compartidos.
- En la seleccion de idiomas se muestra cobertura por videos (X/Y) ademas del total de pistas.
"""

import json
import os
import re
import shutil
import subprocess
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

VIDEO_EXTS = {'.mkv', '.mp4', '.avi'}
SUB_EXTS = {'.srt', '.ass'}
FFPROBE_FIELDS = 'stream=index,codec_type,codec_name:stream_tags=language'
SUB_FILE_RE = re.compile(
    r'^(?P<base>.+)_(?P<lang>[A-Za-z0-9-]+)_sub(?P<idx>\d+)(?P<suffix>_[^.]+)?\.(?P<ext>srt|ass)$',
    flags=re.IGNORECASE,
)
EPISODE_RE = re.compile(r'(?i)\bS\d{1,2}E\d{1,3}\b')
SI_VALUES = {'s', 'si', 'y', 'yes'}

LANG_CODE_MAP = {
    'es': 'spa',
    'es-419': 'spa',
    'es419': 'spa',
    'spa': 'spa',
    'spanish': 'spa',
    'en': 'eng',
    'eng': 'eng',
    'english': 'eng',
    'fr': 'fra',
    'fre': 'fra',
    'fra': 'fra',
    'de': 'deu',
    'ger': 'deu',
    'deu': 'deu',
    'it': 'ita',
    'ita': 'ita',
    'pt': 'por',
    'pt-br': 'por',
    'por': 'por',
    'ja': 'jpn',
    'jp': 'jpn',
    'jpn': 'jpn',
    'ko': 'kor',
    'kor': 'kor',
    'ru': 'rus',
    'rus': 'rus',
    'ar': 'ara',
    'ara': 'ara',
    'tr': 'tur',
    'tur': 'tur',
    'pl': 'pol',
    'pol': 'pol',
    'zh': 'zho',
    'zh-cn': 'zho',
    'zh-tw': 'zho',
    'zho': 'zho',
}

LANG_TITLE_MAP = {
    'spa': 'Espanol (Latinoamerica)',
    'eng': 'English',
    'fra': 'Francais',
    'deu': 'Deutsch',
    'ita': 'Italiano',
    'por': 'Portugues',
    'jpn': 'Japanese',
    'kor': 'Korean',
    'rus': 'Russian',
    'ara': 'Arabic',
    'tur': 'Turkish',
    'pol': 'Polish',
    'zho': 'Chinese',
}


CONSOLE = Console() if RICH_AVAILABLE else None
LANGUAGE_COUNT_HINT = {}
LANGUAGE_VIDEO_HINT = {}
LANGUAGE_TOTAL_VIDEO_HINT = 0
DISCARD_DIR_NAME = "__subs_descartados"

SDH_CUE_RE = re.compile(
    r"(music|song|sfx|sound|applause|laugh|laughter|sigh|gasp|grunt|groan|pant|breath|breathing|"
    r"footsteps|door|knock|ring|phone|wind|rain|thunder|cry|sob|whisper|moan|slap|kiss)",
    flags=re.IGNORECASE,
)
BRACKETED_RE = re.compile(r"^\s*[\[(].{0,120}[\])]\s*$")
ASS_DIALOGUE_RE = re.compile(r"^Dialogue:", flags=re.IGNORECASE)
ASS_TAG_STRIP_RE = re.compile(r"\{[^}]*\}")


def ui_print(message="", style=None):
    if RICH_AVAILABLE:
        CONSOLE.print(message, style=style)
    else:
        print(message)


def ui_status(level, message):
    labels = {
        'ok': ('OK', 'green'),
        'warn': ('WARN', 'yellow'),
        'err': ('ERR', 'red'),
        'run': ('RUN', 'cyan'),
        'auto': ('AUTO', 'magenta'),
        'info': ('INFO', 'blue'),
    }
    label, color = labels.get(level, ('INFO', 'blue'))
    if RICH_AVAILABLE:
        text = Text()
        text.append(f"[{label}] ", style=f"bold {color}")
        text.append(str(message))
        CONSOLE.print(text)
    else:
        print(f"[{label}] {message}")


def ui_section(title):
    if RICH_AVAILABLE:
        CONSOLE.rule(f"[bold cyan]{title}[/bold cyan]")
    else:
        print(f"\n-------- {title} --------")


def ui_title(title, subtitle=None):
    if RICH_AVAILABLE:
        body = title if not subtitle else f"{title}\n[dim]{subtitle}[/dim]"
        CONSOLE.print(Panel.fit(body, border_style="bright_blue"))
    else:
        print(f"\n=== {title} ===")
        if subtitle:
            print(subtitle)


def ask_yes_no(prompt, default=False):
    suffix = '[S/n]' if default else '[s/N]'
    raw = input(f"{prompt} {suffix}: ").strip().lower()
    if not raw:
        return default
    return raw in SI_VALUES


def shorten_name(file_name, reserve=18):
    width = shutil.get_terminal_size((120, 20)).columns
    max_len = max(40, width - reserve)
    if len(file_name) <= max_len:
        return file_name
    head = max(12, (max_len - 3) // 2)
    tail = max(12, max_len - 3 - head)
    return f"{file_name[:head]}...{file_name[-tail:]}"


def find_mkvmerge_exe():
    exe = shutil.which('mkvmerge')
    if exe:
        return exe

    candidates = [
        r'C:\Program Files\MKVToolNix\mkvmerge.exe',
        r'C:\Program Files (x86)\MKVToolNix\mkvmerge.exe',
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def find_mkvextract_exe():
    exe = shutil.which('mkvextract')
    if exe:
        return exe

    candidates = [
        r'C:\Program Files\MKVToolNix\mkvextract.exe',
        r'C:\Program Files (x86)\MKVToolNix\mkvextract.exe',
    ]
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def _run(cmd, *, cwd: Optional[str] = None):
    result = subprocess.run(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return (
        result.returncode,
        result.stdout.decode(errors='replace'),
        result.stderr.decode(errors='replace'),
    )


def _last_error_line(stderr_text):
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    return lines[-1] if lines else 'Sin detalle de error.'


def list_video_files(path):
    # scandir is materially faster than listdir when there are many files.
    out = []
    with os.scandir(path) as it:
        for entry in it:
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name)
            if ext.lower() in VIDEO_EXTS:
                out.append(entry.name)
    out.sort()
    return out


def list_subtitle_files(path):
    out = []
    with os.scandir(path) as it:
        for entry in it:
            if not entry.is_file():
                continue
            _, ext = os.path.splitext(entry.name)
            if ext.lower() in SUB_EXTS:
                out.append(entry.name)
    out.sort()
    return out


@dataclass(frozen=True)
class VideoKey:
    name: str
    size: int
    mtime_ns: int


def _scan_cache_path(folder: str) -> str:
    return os.path.join(folder, ".subs_scan_cache.json")


def _load_scan_cache(folder: str) -> Dict[str, dict]:
    cache_path = _scan_cache_path(folder)
    if not os.path.isfile(cache_path):
        return {}
    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("videos"), dict):
            return data["videos"]
    except Exception:
        return {}
    return {}


def _save_scan_cache(folder: str, videos_cache: Dict[str, dict]) -> None:
    cache_path = _scan_cache_path(folder)
    tmp_path = cache_path + ".tmp"
    payload = {"version": 2, "videos": videos_cache}
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False)
        os.replace(tmp_path, cache_path)
    except Exception:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def _video_key(folder: str, file_name: str) -> Optional[VideoKey]:
    path = os.path.join(folder, file_name)
    try:
        st = os.stat(path)
    except OSError:
        return None
    return VideoKey(name=file_name, size=int(st.st_size), mtime_ns=int(st.st_mtime_ns))


def _ffprobe_subtitle_streams(folder: str, file_name: str) -> Tuple[List[Tuple[str, int, str, str, str]], List[str]]:
    """Return (streams, langs) for one video file."""
    idiomas, streams = set(), []
    code, out, err = _run([
        'ffprobe', '-v', 'error',
        # Make scanning much faster on big files; subtitle track listing should be in headers/metadata.
        # If a weird file fails to expose tracks with these settings, user can disable cache and retry;
        # we keep a conservative probesize but avoid long analyze passes.
        '-analyzeduration', '0',
        '-probesize', '256k',
        '-select_streams', 's',
        '-show_entries', FFPROBE_FIELDS,
        '-of', 'json', os.path.join(folder, file_name)
    ])
    if code != 0:
        ui_status('warn', f"ffprobe fallo en '{shorten_name(file_name)}': {err.strip()}")
        return [], []

    try:
        data = json.loads(out)
        for s in data.get('streams', []):
            if s.get('codec_type') != 'subtitle':
                continue
            idx = s['index']
            codec = s.get('codec_name', 'unknown')
            lang = (s.get('tags', {}) or {}).get('language', 'und')
            idiomas.add(lang)
            streams.append((file_name, int(idx), lang, codec, 'ffprobe'))
    except json.JSONDecodeError:
        ui_status('warn', f"No se pudo leer la salida de ffprobe para '{shorten_name(file_name)}'.")
        return [], []

    return streams, sorted(idiomas)


def _codec_from_mkvmerge_track(track: dict) -> str:
    # mkvmerge JSON varies slightly by version. Prefer codec_id (Matroska codec id).
    codec_id = str((track.get("properties") or {}).get("codec_id") or "")
    codec = str(track.get("codec") or "")
    raw = (codec_id or codec).upper()

    # Common text subtitle codecs in Matroska.
    if "S_TEXT/ASS" in raw or "S_TEXT/SSA" in raw or "ASS" in raw or "SSA" in raw:
        return "ass"
    if "S_TEXT/UTF8" in raw or "SUBRIP" in raw or "SRT" in raw:
        return "subrip"

    # Image-based / other subs we don't translate directly.
    if "S_HDMV/PGS" in raw or "PGS" in raw:
        return "pgs"
    if "S_VOBSUB" in raw or "VOBSUB" in raw:
        return "vobsub"
    if "S_DVBSUB" in raw or "DVBSUB" in raw:
        return "dvbsub"

    return (codec_id or codec or "unknown").lower()


def _mkvmerge_subtitle_streams(mkvmerge_exe: str, folder: str, file_name: str) -> Tuple[List[Tuple[str, int, str, str, str]], List[str]]:
    """Return (streams, langs) for one MKV file using mkvmerge -J (fast)."""
    idiomas, streams = set(), []
    code, out, err = _run([mkvmerge_exe, '-J', os.path.join(folder, file_name)])
    if code != 0:
        ui_status('warn', f"mkvmerge fallo en '{shorten_name(file_name)}': {err.strip()}")
        return [], []

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        ui_status('warn', f"No se pudo leer la salida de mkvmerge para '{shorten_name(file_name)}'.")
        return [], []

    tracks = data.get("tracks", []) if isinstance(data, dict) else []
    for t in tracks:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "subtitles":
            continue
        tid = t.get("id")
        if tid is None:
            continue
        props = t.get("properties") or {}
        lang = str(props.get("language") or "und")
        codec = _codec_from_mkvmerge_track(t)
        idiomas.add(lang)
        streams.append((file_name, int(tid), lang, codec, 'mkvmerge'))

    return streams, sorted(idiomas)


def _summarize_streams_by_language(streams: List[Tuple[str, int, str, str, str]]):
    by_lang = defaultdict(list)
    for file_name, idx, lang, codec, _tool in streams:
        by_lang[lang].append((file_name, idx, codec))
    return by_lang


def _build_scan_coverage(path: str, streams: List[Tuple[str, int, str, str, str]]):
    videos = list_video_files(path)
    by_video = {video_name: [] for video_name in videos}
    by_lang_videos = defaultdict(set)

    for video_name, idx, lang, codec, _tool in streams:
        by_video.setdefault(video_name, []).append((int(idx), str(lang), str(codec)))
        by_lang_videos[str(lang)].add(video_name)

    for items in by_video.values():
        items.sort(key=lambda x: (int(x[0]), str(x[1]).lower(), str(x[2]).lower()))

    return videos, by_video, by_lang_videos


def _print_scan_coverage(videos, by_video, by_lang_videos) -> None:
    if not videos:
        return

    total_videos = len(videos)
    videos_with_subs = sum(1 for v in videos if by_video.get(v))
    videos_without_subs = total_videos - videos_with_subs

    ui_section("Resumen Cobertura Subtitulos")
    ui_print(f"Videos analizados: {total_videos}")
    ui_print(f"Videos con subtitulos: {videos_with_subs}")
    if videos_without_subs:
        ui_print(
            f"Videos sin subtitulos: {videos_without_subs}",
            style="bold yellow" if RICH_AVAILABLE else None,
        )

    if by_lang_videos:
        shared_langs = sorted(lang for lang, vids in by_lang_videos.items() if len(vids) == total_videos)
        if shared_langs:
            ui_print(
                f"Idiomas compartidos por todos: {', '.join(shared_langs)}",
                style="bold green" if RICH_AVAILABLE else None,
            )
        else:
            ui_print(
                "Idiomas compartidos por todos: ninguno",
                style="bold yellow" if RICH_AVAILABLE else None,
            )

        ui_print("Cobertura por idioma:")
        for lang, vids in sorted(by_lang_videos.items(), key=lambda x: (-len(x[1]), x[0].lower())):
            ui_print(f"- {lang}: {len(vids)}/{total_videos} video(s)")

    ui_section("Detalle Subtitulos por Video")
    for video_name in videos:
        tracks = by_video.get(video_name) or []
        if not tracks:
            ui_print(
                f"- {shorten_name(video_name)} -> 0 pista(s)",
                style="yellow" if RICH_AVAILABLE else None,
            )
            continue

        lang_counts = defaultdict(int)
        for _idx, lang, _codec in tracks:
            lang_counts[lang] += 1
        lang_summary = ", ".join(
            f"{lang}x{count}" if count > 1 else lang
            for lang, count in sorted(lang_counts.items(), key=lambda x: x[0].lower())
        )
        ui_print(f"- {shorten_name(video_name)} -> {len(tracks)} pista(s) | idiomas: {lang_summary}")


def _print_duplicate_streams(streams: List[Tuple[str, int, str, str, str]]) -> None:
    """Show per-video languages that have multiple subtitle streams."""
    by_video = defaultdict(list)
    for video_name, idx, lang, codec, _tool in streams:
        by_video[video_name].append((lang, idx, codec))

    duplicates = []
    for video_name, lst in by_video.items():
        per_lang = defaultdict(list)
        for lang, idx, codec in lst:
            per_lang[lang].append((idx, codec))
        for lang, items in per_lang.items():
            if len(items) > 1:
                duplicates.append((video_name, lang, items))

    if not duplicates:
        return

    ui_section("Detalle Pistas Duplicadas")
    ui_print("Videos con mas de 1 pista de subtitulo por idioma:", style="bold yellow" if RICH_AVAILABLE else None)
    for video_name, lang, items in sorted(duplicates, key=lambda x: (x[0].lower(), x[1].lower())):
        codecs = defaultdict(int)
        for _, codec in items:
            codecs[str(codec).lower()] += 1
        codecs_str = ", ".join(f"{k}={v}" for k, v in sorted(codecs.items()))
        ui_print(f"- {shorten_name(video_name)} | {lang}: {len(items)} pista(s) ({codecs_str})")


def scan_directory(path, *, workers=6, use_cache=True):
    """Scan subtitle streams inside videos in `path`.

    Returns (idiomas, streams).
    streams -> list[(video_file, index, lang, codec, tool)]
      tool: 'mkvmerge' (for MKV using mkvmerge IDs) or 'ffprobe' (for ffprobe stream indices)
    """
    idiomas, streams = set(), []
    videos = list_video_files(path)
    if not videos:
        return [], []

    mkvmerge_exe = find_mkvmerge_exe()
    mkvextract_exe = find_mkvextract_exe()

    videos_cache = _load_scan_cache(path) if use_cache else {}
    cache_changed = False

    to_probe: List[Tuple[VideoKey, str]] = []
    for f in videos:
        key = _video_key(path, f)
        if key is None:
            continue
        cached = videos_cache.get(f)
        ext = os.path.splitext(f)[1].lower()
        # If MKVToolNix is available, prefer mkvmerge for MKV even if cache exists from ffprobe runs,
        # so we can later use mkvextract (and it's usually faster to scan too).
        require_mkvtoolnix = (ext == '.mkv' and bool(mkvmerge_exe) and bool(mkvextract_exe))
        if cached and cached.get("size") == key.size and cached.get("mtime_ns") == key.mtime_ns:
            cached_streams = cached.get("streams") or []
            # If cache doesn't carry mkvmerge ids, refresh it when we can.
            if require_mkvtoolnix and any((item.get("tool") or "ffprobe") != "mkvmerge" for item in cached_streams):
                to_probe.append((key, f))
                continue

            for item in cached_streams:
                try:
                    tool = str(item.get("tool") or "ffprobe")
                    streams.append((
                        f,
                        int(item["index"]),
                        str(item.get("lang") or "und"),
                        str(item.get("codec") or "unknown"),
                        tool,
                    ))
                    idiomas.add(str(item.get("lang") or "und"))
                except Exception:
                    # If cache is malformed, just re-probe this file.
                    to_probe.append((key, f))
                    break
        else:
            to_probe.append((key, f))

    if to_probe:
        ui_status("run", f"Escaneando pistas de subtitulos en {len(to_probe)}/{len(videos)} videos...")

        max_workers = max(1, int(workers or 1))
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            def _scan_one(folder: str, file_name: str):
                ext = os.path.splitext(file_name)[1].lower()
                if ext == '.mkv' and mkvmerge_exe and mkvextract_exe:
                    return _mkvmerge_subtitle_streams(mkvmerge_exe, folder, file_name)
                return _ffprobe_subtitle_streams(folder, file_name)

            futs = {ex.submit(_scan_one, path, f): (key, f) for key, f in to_probe}
            for fut in as_completed(futs):
                key, f = futs[fut]
                try:
                    s, langs = fut.result()
                except Exception as exc:
                    ui_status("warn", f"Escaneo fallo en '{shorten_name(f)}': {exc}")
                    continue

                for entry in s:
                    file_name, idx, lang, codec, tool = entry
                    streams.append((file_name, int(idx), lang, codec, tool))
                    idiomas.add(lang)

                if use_cache:
                    videos_cache[f] = {
                        "size": key.size,
                        "mtime_ns": key.mtime_ns,
                        "streams": [
                            {"index": idx, "lang": lang, "codec": codec, "tool": tool}
                            for _fn, idx, lang, codec, tool in s
                        ],
                    }
                    cache_changed = True

    if use_cache and cache_changed:
        _save_scan_cache(path, videos_cache)

    # Normalize streams ordering for stable UI (then extraction groups by video anyway).
    streams.sort(key=lambda x: (x[0].lower(), int(x[1]), str(x[2]).lower(), str(x[3]).lower(), str(x[4]).lower()))
    return sorted(idiomas), streams


def _is_image_subtitle_codec(codec: str) -> bool:
    c = (codec or "").strip().lower()
    return c in {
        "hdmv_pgs_subtitle",
        "pgs",
        "vobsub",
        "dvd_subtitle",
        "dvb_subtitle",
        "dvbsub",
        "xsub",
    }


def _pick_output_for_codec(codec: str) -> Tuple[str, str]:
    c = (codec or "").strip().lower()
    if c == 'ass':
        return '.ass', 'copy'
    if c in {'srt', 'subrip'}:
        return '.srt', 'copy'
    # For other text subtitle formats (e.g. mov_text), attempt to convert to SRT via ffmpeg.
    return '.srt', 'srt'


def extract_subs(path, streams, idiomas_sel):
    ok, fail = 0, 0
    skipped = 0
    errores = defaultdict(list)
    extracted_paths = []

    mkvextract_exe = find_mkvextract_exe()

    # Group by video so we can extract multiple tracks with ONE tool invocation per file.
    by_video = defaultdict(list)
    for file_name, idx, lang, codec, tool in streams:
        if lang in idiomas_sel:
            by_video[file_name].append((int(idx), str(lang), str(codec), str(tool)))

    for file_name, items in sorted(by_video.items(), key=lambda x: x[0].lower()):
        video_path = os.path.join(path, file_name)
        base = os.path.splitext(file_name)[0]
        ext_video = os.path.splitext(file_name)[1].lower()

        # Prefer MKVToolNix extraction for MKV when we have mkvmerge IDs (tool='mkvmerge').
        can_use_mkvextract = (
            ext_video == '.mkv'
            and mkvextract_exe
            and items
            and all(t == 'mkvmerge' for _idx, _lang, _codec, t in items)
        )

        expected = []  # list[(out_path, idx, lang, codec, impl)]
        if can_use_mkvextract:
            # mkvextract cannot "convert"; we only extract text tracks we can translate.
            track_specs = []
            for idx, lang, codec, _tool in sorted(items, key=lambda x: int(x[0])):
                if _is_image_subtitle_codec(codec):
                    skipped += 1
                    ui_status('warn', f"Omitido {lang} | {str(codec).upper()} (imagen/OCR) en '{shorten_name(file_name)}'")
                    continue
                ext, _codec_opt = _pick_output_for_codec(codec)
                if _codec_opt != 'copy':
                    # mkvextract only copies the track as-is; fall back to ffmpeg for conversion codecs.
                    # This mainly matters for non-MKV containers, but keep it safe.
                    skipped += 1
                    ui_status('warn', f"Omitido {lang} | {str(codec).upper()} (no extraible directo con mkvextract) en '{shorten_name(file_name)}'")
                    continue
                out_file = f"{base}_{lang}_sub{idx}{ext}"
                out_path = os.path.join(path, out_file)
                # Use cwd=SUBS_BULK and relative outputs to avoid ambiguity with drive letter colons on Windows.
                track_specs.append(f"{idx}:{out_file}")
                expected.append((out_path, idx, lang, codec, 'mkvextract'))

            if track_specs:
                code, _out, err = _run([mkvextract_exe, 'tracks', file_name, *track_specs], cwd=path)
                if code != 0:
                    # Mark all expected as failed; mkvextract returns a single code for the batch.
                    for out_path, idx, lang, codec, _impl in expected:
                        fail += 1
                        errores[file_name].append((idx, lang, codec))
                        ui_status(
                            'err',
                            (
                                f"Error al extraer {lang} | {str(codec).upper()} de "
                                f"'{shorten_name(file_name)}'. mkvextract: {_last_error_line(err)}"
                            ),
                        )
                else:
                    for out_path, idx, lang, codec, _impl in expected:
                        if os.path.isfile(out_path):
                            ok += 1
                            extracted_paths.append(out_path)
                            ui_status('ok', f"Extraido {lang} | {str(codec).upper():<7} -> {shorten_name(os.path.basename(out_path))}")
                        else:
                            fail += 1
                            errores[file_name].append((idx, lang, codec))
                            ui_status(
                                'err',
                                (
                                    f"Error al extraer {lang} | {str(codec).upper()} de "
                                    f"'{shorten_name(file_name)}'. Detalle: salida no creada."
                                ),
                            )
            continue

        # Fallback: ffmpeg. Build ONE command for the whole file (multiple outputs).
        cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-nostdin', '-y', '-i', video_path]
        for idx, lang, codec, _tool in sorted(items, key=lambda x: int(x[0])):
            if _is_image_subtitle_codec(codec):
                skipped += 1
                ui_status('warn', f"Omitido {lang} | {str(codec).upper()} (imagen/OCR) en '{shorten_name(file_name)}'")
                continue
            ext, codec_opt = _pick_output_for_codec(codec)
            out_file = f"{base}_{lang}_sub{idx}{ext}"
            out_path = os.path.join(path, out_file)
            expected.append((out_path, idx, lang, codec, 'ffmpeg'))
            cmd.extend(['-map', f'0:{idx}', '-c:s', codec_opt, out_path])

        if not expected:
            continue

        code, _out, err = _run(cmd)
        if code != 0:
            # ffmpeg batches; if it fails, report per track with the shared last error line.
            err_line = _last_error_line(err)
            for out_path, idx, lang, codec, _impl in expected:
                fail += 1
                errores[file_name].append((idx, lang, codec))
                ui_status(
                    'err',
                    (
                        f"Error al extraer {lang} | {str(codec).upper()} de "
                        f"'{shorten_name(file_name)}'. ffmpeg: {err_line}"
                    ),
                )
            continue

        for out_path, idx, lang, codec, _impl in expected:
            if os.path.isfile(out_path):
                ok += 1
                extracted_paths.append(out_path)
                ui_status('ok', f"Extraido {lang} | {str(codec).upper():<7} -> {shorten_name(os.path.basename(out_path))}")
            else:
                fail += 1
                errores[file_name].append((idx, lang, codec))
                ui_status(
                    'err',
                    (
                        f"Error al extraer {lang} | {str(codec).upper()} de "
                        f"'{shorten_name(file_name)}'. Detalle: salida no creada."
                    ),
                )

    ui_section("Resumen Extraccion")
    ui_print(f"Subtitulos extraidos correctamente: {ok}", style="bold green" if RICH_AVAILABLE else None)
    if skipped:
        ui_print(f"Subtitulos omitidos (imagen/OCR): {skipped}", style="bold yellow" if RICH_AVAILABLE else None)
    if fail:
        ui_print(f"Subtitulos con error: {fail}", style="bold red" if RICH_AVAILABLE else None)
        for f, lst in errores.items():
            detalles = ', '.join(f"{i}:{l}" for i, l, _ in lst)
            ui_print(f"  - {shorten_name(f)}: {detalles}")

    if extracted_paths:
        ui_section("Detalle Subtitulos Extraidos")
        _print_extracted_summary(path, extracted_paths)

    return extracted_paths


def seleccionar_modo():
    ui_title("Modo de trabajo", "Selecciona que quieres hacer")
    ui_print(" 1. Extraer subtitulos", style="cyan" if RICH_AVAILABLE else None)
    ui_print(" 2. Muxear subtitulos traducidos", style="cyan" if RICH_AVAILABLE else None)
    ui_print(" 3. Extraer y luego muxear", style="cyan" if RICH_AVAILABLE else None)
    raw = input("Elige una opcion [1-3] (default 3): ").strip()
    if raw in {'1', '2', '3'}:
        return int(raw)
    return 3


def seleccionar_idiomas(idiomas):
    ui_title("Idiomas encontrados")
    # `idiomas` is already sorted; show counts if available via global hint.
    for i, lang in enumerate(idiomas, 1):
        track_hint = LANGUAGE_COUNT_HINT.get(lang)
        video_hint = LANGUAGE_VIDEO_HINT.get(lang)
        if (
            track_hint is not None
            and video_hint is not None
            and LANGUAGE_TOTAL_VIDEO_HINT
        ):
            ui_print(
                f" {i}. {lang} ({track_hint} pista(s), {video_hint}/{LANGUAGE_TOTAL_VIDEO_HINT} video(s))",
                style="cyan" if RICH_AVAILABLE else None,
            )
        elif track_hint is not None:
            ui_print(f" {i}. {lang} ({track_hint} pista(s))", style="cyan" if RICH_AVAILABLE else None)
        else:
            ui_print(f" {i}. {lang}", style="cyan" if RICH_AVAILABLE else None)
    ui_print(" Tip: escribe 'all' para seleccionar todos.", style="dim" if RICH_AVAILABLE else None)
    sel = input("Elige los numeros de los idiomas a extraer (ej. 1,3) o 'all': ").strip().lower()
    if sel in {"all", "*"}:
        return list(idiomas)
    nums = [int(x.strip()) for x in sel.split(',') if x.strip().isdigit()]
    return [idiomas[i - 1] for i in nums if 0 < i <= len(idiomas)]


def _safe_read_lines(path: str) -> Iterable[str]:
    # Some subs have weird encodings; be permissive.
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            yield line


def _count_ass_dialogue_lines(path: str) -> int:
    count = 0
    for line in _safe_read_lines(path):
        if line.startswith("Dialogue:"):
            count += 1
    return count


def _count_srt_entries(path: str) -> int:
    # Counting timestamp lines is fast and robust.
    count = 0
    for line in _safe_read_lines(path):
        if "-->" in line:
            count += 1
    return count


def _subtitle_file_metrics(path: str) -> Tuple[int, int]:
    """Return (entries/dialogue, bytes)."""
    try:
        size = os.path.getsize(path)
    except OSError:
        size = 0
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".ass":
        return _count_ass_dialogue_lines(path), size
    if ext == ".srt":
        return _count_srt_entries(path), size
    return 0, size


def _extract_file_lang_and_base(file_name: str) -> Tuple[str, str]:
    """Return (lang, base_stem) for extracted subtitle files."""
    m = SUB_FILE_RE.match(file_name)
    if m:
        return (m.group("lang"), m.group("base"))
    stem = os.path.splitext(file_name)[0]
    return ("und", stem)


def _iter_ass_entries(path: str) -> Iterable[str]:
    # ASS has one subtitle "entry" per Dialogue: line, where the payload is the last CSV field.
    for line in _safe_read_lines(path):
        if not ASS_DIALOGUE_RE.match(line):
            continue
        parts = line.split(",", 9)
        if len(parts) < 10:
            continue
        text = parts[9].rstrip("\r\n")
        text = ASS_TAG_STRIP_RE.sub("", text)
        text = text.replace("\\N", " ").replace("\\n", " ").replace("\\h", " ")
        yield text.strip()


def _iter_srt_entries(path: str) -> Iterable[str]:
    buf: List[str] = []
    in_payload = False
    for raw in _safe_read_lines(path):
        line = raw.rstrip("\r\n")
        if not line.strip():
            if buf:
                text = " ".join(s.strip() for s in buf if s.strip()).strip()
                if text:
                    yield text
                buf = []
            in_payload = False
            continue
        if "-->" in line:
            in_payload = True
            continue
        if not in_payload:
            # Skip numeric counter lines.
            continue
        buf.append(line)
    if buf:
        text = " ".join(s.strip() for s in buf if s.strip()).strip()
        if text:
            yield text


def _subtitle_content_metrics(path: str) -> dict:
    """Cheap heuristics to avoid SDH/full subs when a cleaner track exists."""
    _, ext = os.path.splitext(path)
    ext = ext.lower()
    if ext == ".ass":
        entries_iter = _iter_ass_entries(path)
    else:
        entries_iter = _iter_srt_entries(path)

    entries = 0
    sdh_like = 0
    music_like = 0
    empty = 0

    for text in entries_iter:
        entries += 1
        t = (text or "").strip()
        if not t:
            empty += 1
            continue
        if "♪" in t or "♫" in t:
            music_like += 1
        tl = t.lower()
        # Typical SDH cues are bracketed and contain SFX keywords: [door opens], (laughs), etc.
        if BRACKETED_RE.match(t) and SDH_CUE_RE.search(tl):
            sdh_like += 1
        elif SDH_CUE_RE.search(tl) and len(tl) <= 40:
            # Some tracks use plain "SFX:" lines without brackets.
            sdh_like += 1

    dialogue = max(0, entries - sdh_like - music_like - empty)

    # Strongly avoid "signs/forced only" tracks unless it's the only option.
    tiny_penalty = 1000 if entries > 0 and entries < 40 else 0

    score = (dialogue * 1.0) - (sdh_like * 1.5) - (music_like * 1.5) - tiny_penalty

    return {
        "entries": entries,
        "dialogue": dialogue,
        "sdh_like": sdh_like,
        "music_like": music_like,
        "empty": empty,
        "score": score,
        "bytes": os.path.getsize(path) if os.path.isfile(path) else 0,
    }


def _ensure_discard_dir(folder: str) -> str:
    d = os.path.join(folder, DISCARD_DIR_NAME)
    os.makedirs(d, exist_ok=True)
    return d


def _move_to_discard(folder: str, file_path: str) -> str:
    discard_dir = _ensure_discard_dir(folder)
    name = os.path.basename(file_path)
    dest = os.path.join(discard_dir, name)
    if os.path.abspath(file_path) == os.path.abspath(dest):
        return dest
    if os.path.exists(dest):
        base, ext = os.path.splitext(name)
        i = 1
        while True:
            alt = os.path.join(discard_dir, f"{base}__dup{i}{ext}")
            if not os.path.exists(alt):
                dest = alt
                break
            i += 1
    os.replace(file_path, dest)
    return dest


def auto_select_best_subs(folder: str, extracted_paths: List[str], auto: bool) -> None:
    """When multiple extracted subtitle files share the same (base, lang), keep the best and discard the rest."""
    groups = defaultdict(list)
    for p in extracted_paths:
        if not os.path.isfile(p):
            continue
        name = os.path.basename(p)
        lang, base = _extract_file_lang_and_base(name)
        groups[(base, lang)].append(p)

    dup_groups = {k: v for k, v in groups.items() if len(v) > 1}
    if not dup_groups:
        return

    if not auto:
        if not ask_yes_no(
            "Detecte subtitulos duplicados por idioma. Conservar solo el mejor y mover el resto a respaldo?",
            default=True,
        ):
            ui_status("info", "Se omitio la seleccion automatica de subtitulos duplicados.")
            return

    ui_section("Auto Seleccion Subtitulos")
    ui_status("run", f"Analizando {sum(len(v) for v in dup_groups.values())} archivo(s) en {len(dup_groups)} grupo(s)...")

    kept = 0
    moved = 0
    for (base, lang), paths in sorted(dup_groups.items()):
        metrics = [(p, _subtitle_content_metrics(p)) for p in paths]
        # Pick the best by score, then by dialogue count, then by entries.
        metrics.sort(key=lambda x: (x[1]["score"], x[1]["dialogue"], x[1]["entries"], x[1]["bytes"]), reverse=True)
        best_path, best_m = metrics[0]
        ui_print(f"- {shorten_name(base)} | {lang}: conservar -> {shorten_name(os.path.basename(best_path))}", style="bold cyan" if RICH_AVAILABLE else None)
        ui_print(f"    score={best_m['score']:.1f} entries={best_m['entries']} dialogue={best_m['dialogue']} sdh={best_m['sdh_like']} music={best_m['music_like']} bytes={best_m['bytes']}")
        kept += 1

        for p, m in metrics[1:]:
            dest = _move_to_discard(folder, p)
            moved += 1
            ui_print(
                f"    mover -> {shorten_name(os.path.basename(dest))} | score={m['score']:.1f} entries={m['entries']} dialogue={m['dialogue']} sdh={m['sdh_like']} music={m['music_like']} bytes={m['bytes']}",
            )

    ui_status("ok", f"Seleccion completada. grupos={kept}, movidos={moved}. Respaldo: {os.path.join(folder, DISCARD_DIR_NAME)}")


def _print_extracted_summary(folder: str, extracted_paths: List[str]) -> None:
    groups = defaultdict(list)
    for p in extracted_paths:
        name = os.path.basename(p)
        lang, base = _extract_file_lang_and_base(name)
        groups[(base, lang)].append(p)

    # Stable-ish order.
    for (base, lang) in sorted(groups.keys()):
        items = groups[(base, lang)]
        ui_print(f"- {shorten_name(base)} | {lang} -> {len(items)} archivo(s)", style="bold cyan" if RICH_AVAILABLE else None)
        # Show the details so user can see duplicates (size/lines differ).
        for p in sorted(items):
            entries, size = _subtitle_file_metrics(p)
            ui_print(f"    {shorten_name(os.path.basename(p))} | {size} bytes | entradas/lineas: {entries}")


def es_video_generado(file_name):
    stem = os.path.splitext(file_name)[0].lower()
    return stem.endswith('_muxed') or '.bak' in stem or '__mux_tmp' in stem


def _normalize_lang_token(token):
    t = token.strip().lower().replace('_', '-').replace('.', '-')
    return LANG_CODE_MAP.get(t, t)


def _parse_language_tokens(stem_lower):
    return re.findall(r'[a-z0-9]+(?:-[a-z0-9]+)?', stem_lower)


def es_subtitulo_traducido(file_name):
    stem = os.path.splitext(file_name)[0].lower()

    if re.search(r'_sub\d+_[a-z0-9-]+$', stem):
        return True

    tokens = _parse_language_tokens(stem)
    if len(tokens) < 2:
        return False

    last = _normalize_lang_token(tokens[-1])
    if last not in LANG_TITLE_MAP and last not in {'spa', 'eng'}:
        return False

    prev_lang = any(_normalize_lang_token(tok) in LANG_TITLE_MAP for tok in tokens[:-1])
    return prev_lang


def _episode_key(stem):
    m = EPISODE_RE.search(stem)
    return m.group(0).lower() if m else ''


def _token_set(stem):
    return {t for t in re.findall(r'[a-z0-9]+', stem.lower()) if t}


def _build_video_infos(videos):
    infos = []
    for name in videos:
        stem = os.path.splitext(name)[0]
        infos.append({
            'name': name,
            'stem': stem,
            'stem_lower': stem.lower(),
            'episode': _episode_key(stem),
            'tokens': _token_set(stem),
        })
    return infos


def _score_sub_to_video(sub_stem_lower, sub_episode, sub_tokens, info):
    score = 0

    if info['stem_lower'] in sub_stem_lower or sub_stem_lower in info['stem_lower']:
        score += 100

    if sub_episode and info['episode'] and sub_episode == info['episode']:
        score += 65

    overlap = len(sub_tokens & info['tokens'])
    if overlap:
        denom = max(1, min(len(sub_tokens), len(info['tokens'])))
        score += int((overlap / float(denom)) * 40)

    return score


def _match_sub_to_video(file_name, video_infos):
    sub_stem = os.path.splitext(file_name)[0]
    sub_stem_lower = sub_stem.lower()
    sub_episode = _episode_key(sub_stem)
    sub_tokens = _token_set(sub_stem)

    ranked = []
    for info in video_infos:
        score = _score_sub_to_video(sub_stem_lower, sub_episode, sub_tokens, info)
        if score > 0:
            ranked.append((score, info['name']))

    if not ranked:
        return None

    ranked.sort(reverse=True)
    top_score, top_name = ranked[0]
    second_score = ranked[1][0] if len(ranked) > 1 else -1

    if top_score < 55:
        return None
    if second_score >= 0 and (top_score - second_score) < 12:
        return None

    return top_name


def detectar_subs_traducidos(path):
    videos = list_video_files(path)
    videos_principales = [v for v in videos if not es_video_generado(v)]
    videos_ref = videos_principales if videos_principales else videos
    video_infos = _build_video_infos(videos_ref)
    video_by_stem = {os.path.splitext(v)[0]: v for v in videos_ref}

    por_video = defaultdict(list)
    sin_video = []

    for file_name in list_subtitle_files(path):
        if not es_subtitulo_traducido(file_name):
            continue

        m = SUB_FILE_RE.match(file_name)
        if m:
            suffix = (m.group('suffix') or '').strip()
            if not suffix:
                continue
            base = m.group('base')
            video_name = video_by_stem.get(base)
            if video_name:
                por_video[video_name].append(file_name)
                continue

        video_name = _match_sub_to_video(file_name, video_infos)
        if video_name:
            por_video[video_name].append(file_name)
        elif len(videos_ref) == 1:
            por_video[videos_ref[0]].append(file_name)
        else:
            sin_video.append(file_name)

    for video_name, subs in por_video.items():
        por_video[video_name] = sorted(set(subs))

    return por_video, sorted(set(sin_video))


def _extract_lang_token_from_name(sub_name):
    stem = os.path.splitext(sub_name)[0].lower()

    m = SUB_FILE_RE.match(sub_name)
    if m and m.group('suffix'):
        suffix = m.group('suffix').lower().lstrip('_')
        if suffix.startswith('es-419') or suffix.startswith('es_419'):
            return 'es-419'
        suffix_tokens = _parse_language_tokens(suffix)
        for tok in reversed(suffix_tokens):
            norm = _normalize_lang_token(tok)
            if norm in LANG_TITLE_MAP:
                return tok

    if 'es-419' in stem or 'es_419' in stem:
        return 'es-419'

    tokens = _parse_language_tokens(stem)
    for tok in reversed(tokens):
        norm = _normalize_lang_token(tok)
        if norm in LANG_TITLE_MAP:
            return tok

    return 'und'


def inferir_metadata_subtitulo(sub_name):
    token = _extract_lang_token_from_name(sub_name)
    iso3 = _normalize_lang_token(token)
    if iso3 not in LANG_TITLE_MAP:
        return 'und', 'Subtitle [Traducido]', False

    if token in {'es-419', 'es_419'}:
        title = 'Espanol (Latinoamerica) [Traducido]'
    else:
        title = f"{LANG_TITLE_MAP.get(iso3, 'Subtitle')} [Traducido]"

    return iso3, title, iso3 == 'spa'


def _is_spanish_tag(tag_value):
    value = (tag_value or '').strip().lower()
    if not value:
        return False
    if value in {'es', 'es-419', 'spa'}:
        return True
    return _normalize_lang_token(value) == 'spa'


def get_subtitle_streams(video_path):
    code, out, err = _run([
        'ffprobe',
        '-v', 'error',
        '-select_streams', 's',
        '-show_entries', 'stream=index:stream_tags=language,title',
        '-of', 'json',
        video_path,
    ])
    if code != 0:
        ui_status(
            'warn',
            (
                "No se pudo leer subtitulos existentes de "
                f"'{shorten_name(os.path.basename(video_path))}': {err.strip()}"
            ),
        )
        return []

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []

    streams = []
    for s in data.get('streams', []):
        tags = s.get('tags') or {}
        streams.append({
            'index': s.get('index'),
            'language': tags.get('language', ''),
            'title': tags.get('title', ''),
        })
    return streams


def seleccionar_video_para_sub(path, sub_name):
    videos = [v for v in list_video_files(path) if not es_video_generado(v)]
    if not videos:
        return None
    if len(videos) == 1:
        return videos[0]

    ui_status('warn', f"No se pudo asociar automatico: {shorten_name(sub_name)}")
    ui_print("Selecciona el video destino:", style="bold yellow" if RICH_AVAILABLE else None)
    for i, v in enumerate(videos, 1):
        ui_print(f" {i}. {shorten_name(v)}", style="cyan" if RICH_AVAILABLE else None)
    raw = input("Numero de video (Enter = omitir): ").strip()
    if not raw or not raw.isdigit():
        return None
    idx = int(raw)
    if 1 <= idx <= len(videos):
        return videos[idx - 1]
    return None


def _build_backup_path(path, video_name):
    base, ext = os.path.splitext(video_name)
    candidate = os.path.join(path, f"{base}.bak{ext}")
    if not os.path.exists(candidate):
        return candidate

    i = 2
    while True:
        candidate = os.path.join(path, f"{base}.bak{i}{ext}")
        if not os.path.exists(candidate):
            return candidate
        i += 1


def _reemplazar_original_con_tmp(original_path, tmp_output_path, crear_backup):
    base_name = os.path.basename(original_path)
    backup_path = None

    try:
        if crear_backup:
            backup_path = _build_backup_path(os.path.dirname(original_path), base_name)
            os.replace(original_path, backup_path)
        else:
            os.remove(original_path)

        os.replace(tmp_output_path, original_path)
        return True, backup_path, None
    except OSError as exc:
        if os.path.exists(tmp_output_path) and not os.path.exists(original_path):
            try:
                os.replace(tmp_output_path, original_path)
            except OSError:
                pass

        if backup_path and os.path.exists(backup_path) and not os.path.exists(original_path):
            try:
                os.replace(backup_path, original_path)
            except OSError:
                pass

        return False, backup_path, exc


def get_mkvmerge_track_info(mkvmerge_exe, video_path):
    code, out, err = _run([mkvmerge_exe, '-J', video_path])
    if code != 0:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return []
    return data.get('tracks', []) if isinstance(data, dict) else []


def build_mux_cmd_mkvmerge(path, video_name, sub_files, out_path, mkvmerge_exe):
    video_path = os.path.join(path, video_name)
    tracks = get_mkvmerge_track_info(mkvmerge_exe, video_path)
    subtitle_tracks = [t for t in tracks if t.get('type') == 'subtitles']

    cmd = [mkvmerge_exe, '-o', out_path]

    first_existing_spanish_tid = None
    for track in subtitle_tracks:
        tid = track.get('id')
        if tid is None:
            continue
        cmd.extend(['--default-track-flag', f'{tid}:0'])
        props = track.get('properties') or {}
        lang = props.get('language') or ''
        if first_existing_spanish_tid is None and _is_spanish_tag(str(lang)):
            first_existing_spanish_tid = tid

    spanish_new_default = None
    for i, sub_file in enumerate(sub_files):
        _, _, is_spanish = inferir_metadata_subtitulo(sub_file)
        if is_spanish:
            spanish_new_default = i
            break

    if spanish_new_default is None and first_existing_spanish_tid is not None:
        cmd.extend(['--default-track-flag', f'{first_existing_spanish_tid}:1'])

    cmd.append(video_path)

    for i, sub_file in enumerate(sub_files):
        lang_code, title, _ = inferir_metadata_subtitulo(sub_file)
        default_flag = '1' if spanish_new_default == i else '0'
        cmd.extend([
            '--language', f'0:{lang_code}',
            '--track-name', f'0:{title}',
            '--default-track-flag', f'0:{default_flag}',
            os.path.join(path, sub_file),
        ])

    return cmd


def build_mux_cmd(path, video_name, sub_files, out_path):
    video_path = os.path.join(path, video_name)
    existing_subs = get_subtitle_streams(video_path)
    existing_count = len(existing_subs)

    cmd = ['ffmpeg', '-y', '-i', video_path]
    for sub_file in sub_files:
        cmd.extend(['-i', os.path.join(path, sub_file)])

    cmd.extend(['-map', '0'])
    for i in range(len(sub_files)):
        cmd.extend(['-map', f'{i + 1}:0'])

    # Limpia defaults previos y define uno nuevo.
    cmd.extend(['-disposition:s', '0'])

    spanish_output_idx = None

    for i, sub_file in enumerate(sub_files):
        out_sub_idx = existing_count + i
        lang_code, title, is_spanish = inferir_metadata_subtitulo(sub_file)

        cmd.extend([f'-metadata:s:s:{out_sub_idx}', f'language={lang_code}'])
        cmd.extend([f'-metadata:s:s:{out_sub_idx}', f'title={title}'])

        if is_spanish and spanish_output_idx is None:
            spanish_output_idx = out_sub_idx

    if spanish_output_idx is None:
        for idx, stream in enumerate(existing_subs):
            if _is_spanish_tag(stream.get('language', '')):
                spanish_output_idx = idx
                break

    if spanish_output_idx is not None:
        cmd.extend([f'-disposition:s:{spanish_output_idx}', 'default'])

    cmd.extend(['-c', 'copy', out_path])
    return cmd


def muxear_subs_traducidos(path, reemplazar_original=False, crear_backup=True):
    por_video, sin_video = detectar_subs_traducidos(path)
    mkvmerge_exe = find_mkvmerge_exe()

    if sin_video and ask_yes_no("Se detectaron subtitulos sin match. Quieres asignarlos manualmente?", default=True):
        restantes = []
        for sub_name in sin_video:
            selected_video = seleccionar_video_para_sub(path, sub_name)
            if selected_video:
                por_video[selected_video].append(sub_name)
            else:
                restantes.append(sub_name)
        sin_video = restantes

    if sin_video:
        ui_status('warn', "Subtitulos traducidos sin video base detectado:")
        for s in sin_video:
            ui_print(f"  - {shorten_name(s)}", style="yellow" if RICH_AVAILABLE else None)

    if not por_video:
        ui_status('info', "No se detectaron subtitulos traducidos para muxear.")
        ui_print("Tip: se esperan nombres tipo *_subX_es-419.srt o *_eng_es-419.ass", style="dim")
        return []

    muxeados = []
    ok, fail = 0, 0
    ui_status('run', "Iniciando mux de subtitulos traducidos...")

    for video_name, sub_files in sorted(por_video.items()):
        sub_files = sorted(set(sub_files))
        if not sub_files:
            continue

        base = os.path.splitext(video_name)[0]
        ext_video = os.path.splitext(video_name)[1].lower()
        replace_this_video = reemplazar_original and ext_video == '.mkv'

        if replace_this_video:
            out_name = f"{base}__mux_tmp.mkv"
        else:
            out_name = f"{base}_muxed.mkv"

        out_path = os.path.join(path, out_name)

        if reemplazar_original and ext_video != '.mkv':
            ui_status(
                'warn',
                (
                    f"'{shorten_name(video_name)}' no es MKV; se crea salida separada "
                    f"({shorten_name(out_name)})."
                ),
            )

        if ext_video == '.mkv' and mkvmerge_exe:
            cmd = build_mux_cmd_mkvmerge(path, video_name, sub_files, out_path, mkvmerge_exe)
            mux_tool_name = 'mkvmerge'
        else:
            cmd = build_mux_cmd(path, video_name, sub_files, out_path)
            mux_tool_name = 'ffmpeg'

        code, _, err = _run(cmd)

        if code == 0 and os.path.isfile(out_path):
            final_label = out_name
            if replace_this_video:
                ok_replace, backup_path, exc = _reemplazar_original_con_tmp(
                    os.path.join(path, video_name), out_path, crear_backup
                )
                if not ok_replace:
                    fail += 1
                    ui_status(
                        'err',
                        (
                            "Mux generado pero no se pudo reemplazar "
                            f"'{shorten_name(video_name)}'. Detalle: {exc}"
                        ),
                    )
                    continue
                final_label = video_name
                if backup_path:
                    ui_status('ok', f"Backup original: {shorten_name(os.path.basename(backup_path))}")

            ok += 1
            muxeados.extend(os.path.join(path, s) for s in sub_files)
            ui_status(
                'ok',
                (
                    f"Mux OK: {shorten_name(video_name)} + {len(sub_files)} "
                    f"subtitulo(s) -> {shorten_name(final_label)}"
                ),
            )
        else:
            fail += 1
            ui_status(
                'err',
                (
                    f"Error al muxear '{shorten_name(video_name)}' ({mux_tool_name}). "
                    f"Detalle: {_last_error_line(err)}"
                ),
            )

    ui_section("Resumen Mux")
    ui_print(f"Videos muxeados correctamente: {ok}", style="bold green" if RICH_AVAILABLE else None)
    if fail:
        ui_print(f"Videos con error en mux: {fail}", style="bold red" if RICH_AVAILABLE else None)

    return sorted(set(muxeados))


def preguntar_borrado_subs(sub_paths, auto_delete=False, auto_delete_originales=False):
    if not sub_paths:
        return

    ui_title("Subtitulos traducidos usados en mux")
    for sub_path in sub_paths:
        ui_print(f"  - {shorten_name(os.path.basename(sub_path))}", style="cyan" if RICH_AVAILABLE else None)

    if auto_delete:
        ui_status('auto', "Eliminacion de subtitulos muxeados: SI")
    elif not ask_yes_no("\nQuieres eliminar estos subtitulos ya muxeados?", default=False):
        ui_status('info', "Se conservaron los subtitulos.")
        return

    originales_relacionados = detectar_subs_originales_relacionados(sub_paths)
    a_borrar = list(sub_paths)
    eliminar_originales = False
    if originales_relacionados:
        ui_title("Subtitulos originales relacionados detectados")
        for sub_path in originales_relacionados:
            ui_print(f"  - {shorten_name(os.path.basename(sub_path))}", style="cyan" if RICH_AVAILABLE else None)
        if auto_delete_originales:
            eliminar_originales = True
            ui_status('auto', "Eliminacion de subtitulos originales relacionados: SI")
        else:
            eliminar_originales = ask_yes_no(
                "Quieres eliminar tambien estos subtitulos originales?",
                default=False,
            )
        if eliminar_originales:
            a_borrar.extend(originales_relacionados)

    borrados, errores = 0, 0
    for sub_path in sorted(set(a_borrar)):
        try:
            os.remove(sub_path)
            borrados += 1
        except OSError as exc:
            errores += 1
            ui_status('warn', f"No se pudo borrar '{shorten_name(os.path.basename(sub_path))}': {exc}")

    ui_status('ok', f"Subtitulos borrados: {borrados}")
    if originales_relacionados and not eliminar_originales:
        ui_status('info', "Se conservaron los subtitulos originales relacionados.")
    if errores:
        ui_status('warn', f"Errores al borrar: {errores}")


def detectar_subs_originales_relacionados(sub_paths):
    relacionados = set()
    for sub_path in sub_paths:
        folder = os.path.dirname(sub_path)
        file_name = os.path.basename(sub_path)
        stem, ext = os.path.splitext(file_name)
        candidatos = set()

        m = SUB_FILE_RE.match(file_name)
        if m and m.group('suffix'):
            candidatos.add(
                f"{m.group('base')}_{m.group('lang')}_sub{m.group('idx')}{ext}"
            )

        m_last = re.match(r'^(?P<prefix>.+)_(?P<dest>[A-Za-z0-9_-]+)$', stem)
        if m_last:
            dest_norm = _normalize_lang_token(m_last.group('dest'))
            if dest_norm in LANG_TITLE_MAP:
                candidatos.add(f"{m_last.group('prefix')}{ext}")

        for cand in candidatos:
            cand_path = os.path.join(folder, cand)
            if cand_path == sub_path:
                continue
            if os.path.isfile(cand_path):
                relacionados.add(cand_path)

    return sorted(relacionados)


def main():
    parser = argparse.ArgumentParser(
        description="Gestiona extraccion y mux de subtitulos en SUBS_BULK."
    )
    parser.add_argument(
        "--bulk-dir",
        default=None,
        help="Carpeta de trabajo (default: ./SUBS_BULK junto al script)",
    )
    parser.add_argument(
        "--scan-workers",
        type=int,
        default=6,
        help="Hilos para escaneo (ffprobe/mkvmerge) (default: 6)",
    )
    parser.add_argument(
        "--no-scan-cache",
        action="store_true",
        help="Desactiva cache del escaneo",
    )
    parser.add_argument(
        "--no-auto-select-subs",
        action="store_true",
        help="Desactiva la seleccion automatica de duplicados tras la extraccion",
    )
    parser.add_argument(
        '--si',
        action='store_true',
        help=(
            "Modo semi-automatico para mux: reemplaza original, sin backup, "
            "borra subtitulos muxeados y originales relacionados sin preguntar."
        ),
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    bulk_dir = args.bulk_dir or os.path.join(base_dir, 'SUBS_BULK')
    os.makedirs(bulk_dir, exist_ok=True)

    ui_title("Gestor de Subtitulos", f"Carpeta de trabajo: {bulk_dir}")
    modo = seleccionar_modo()

    if modo in {1, 3}:
        idiomas, streams = scan_directory(
            bulk_dir,
            workers=args.scan_workers,
            use_cache=not args.no_scan_cache,
        )

        global LANGUAGE_COUNT_HINT, LANGUAGE_VIDEO_HINT, LANGUAGE_TOTAL_VIDEO_HINT
        LANGUAGE_COUNT_HINT = {lang: len(items) for lang, items in _summarize_streams_by_language(streams).items()}
        scan_videos, scan_by_video, scan_by_lang_videos = _build_scan_coverage(bulk_dir, streams)
        LANGUAGE_TOTAL_VIDEO_HINT = len(scan_videos)
        LANGUAGE_VIDEO_HINT = {lang: len(videos_set) for lang, videos_set in scan_by_lang_videos.items()}
        if not streams:
            ui_status('info', "No se detectaron pistas de subtitulos en videos dentro de SUBS_BULK.")
        else:
            _print_scan_coverage(scan_videos, scan_by_video, scan_by_lang_videos)
            _print_duplicate_streams(streams)
            idiomas_seleccionados = seleccionar_idiomas(idiomas)
            if not idiomas_seleccionados:
                ui_status('info', "No se selecciono ningun idioma para extraer.")
            else:
                ui_status('run', f"Iniciando extraccion para: {', '.join(idiomas_seleccionados)}")
                extracted_paths = extract_subs(bulk_dir, streams, idiomas_seleccionados)
                if extracted_paths and not args.no_auto_select_subs:
                    auto_select_best_subs(bulk_dir, extracted_paths, auto=args.si)

    muxed_sub_paths = []
    if modo in {2, 3}:
        if args.si:
            reemplazar = True
            backup = False
            ui_status('auto', "Reemplazar original: SI")
            ui_status('auto', "Crear backup: NO")
        else:
            reemplazar = ask_yes_no(
                "\nQuieres reemplazar el archivo de video original al finalizar el mux?",
                default=False,
            )
            backup = True
            if reemplazar:
                backup = ask_yes_no("Crear backup del original antes de reemplazar?", default=True)

        muxed_sub_paths = muxear_subs_traducidos(
            bulk_dir,
            reemplazar_original=reemplazar,
            crear_backup=backup,
        )

    preguntar_borrado_subs(
        muxed_sub_paths,
        auto_delete=args.si and modo in {2, 3},
        auto_delete_originales=args.si and modo in {2, 3},
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
