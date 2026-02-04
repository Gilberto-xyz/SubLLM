#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
GESTIONAR_subs_v1.py
Version 3.0 - 04-feb-2026

- Usa SUBS_BULK como carpeta de trabajo para videos y subtitulos.
- Extrae subtitulos de los videos detectados.
- Muxea subtitulos traducidos, fija metadatos y marca espanol como default.
- Puede reemplazar el video original (con backup opcional).
- Al final pregunta si deseas borrar subtitulos ya muxeados.
"""

import json
import os
import re
import shutil
import subprocess
import argparse
from collections import defaultdict

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


def _run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return (
        result.returncode,
        result.stdout.decode(errors='replace'),
        result.stderr.decode(errors='replace'),
    )


def _last_error_line(stderr_text):
    lines = [line.strip() for line in stderr_text.splitlines() if line.strip()]
    return lines[-1] if lines else 'Sin detalle de error.'


def list_video_files(path):
    return sorted(
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in VIDEO_EXTS
    )


def list_subtitle_files(path):
    return sorted(
        f for f in os.listdir(path)
        if os.path.splitext(f)[1].lower() in SUB_EXTS
    )


def scan_directory(path):
    idiomas, streams = set(), []

    for f in list_video_files(path):
        code, out, err = _run([
            'ffprobe', '-v', 'error',
            '-select_streams', 's',
            '-show_entries', FFPROBE_FIELDS,
            '-of', 'json', os.path.join(path, f)
        ])
        if code != 0:
            print(f"[WARN] ffprobe fallo en '{f}': {err.strip()}")
            continue

        try:
            data = json.loads(out)
            for s in data.get('streams', []):
                if s.get('codec_type') != 'subtitle':
                    continue
                idx = s['index']
                codec = s.get('codec_name', 'unknown')
                lang = s.get('tags', {}).get('language', 'und')
                idiomas.add(lang)
                streams.append((f, idx, lang, codec))
        except json.JSONDecodeError:
            print(f"[WARN] No se pudo leer la salida de ffprobe para '{f}'.")

    return sorted(idiomas), streams


def extract_subs(path, streams, idiomas_sel):
    ok, fail = 0, 0
    errores = defaultdict(list)

    for file_name, idx, lang, codec in streams:
        if lang not in idiomas_sel:
            continue

        base = os.path.splitext(file_name)[0]

        if codec == 'ass':
            ext, codec_opt = '.ass', 'copy'
        elif codec in {'srt', 'subrip'}:
            ext, codec_opt = '.srt', 'copy'
        else:
            ext, codec_opt = '.srt', 'srt'

        out_file = f"{base}_{lang}_sub{idx}{ext}"
        out_path = os.path.join(path, out_file)

        code, _, err = _run([
            'ffmpeg', '-y', '-i', os.path.join(path, file_name),
            '-map', f'0:{idx}', '-c:s', codec_opt, out_path
        ])

        if code == 0 and os.path.isfile(out_path):
            ok += 1
            print(f"[OK] Extraido {lang} | {codec.upper():<7} -> {out_file}")
        else:
            fail += 1
            errores[file_name].append((idx, lang, codec))
            print(
                f"[ERR] Error al extraer {lang} | {codec.upper()} de '{file_name}'.\n"
                f"   ffmpeg: {_last_error_line(err)}"
            )

    print("\n-------- Resumen --------")
    print(f"Subtitulos extraidos correctamente: {ok}")
    if fail:
        print(f"Subtitulos con error: {fail}")
        for f, lst in errores.items():
            detalles = ', '.join(f"{i}:{l}" for i, l, _ in lst)
            print(f"  - {f}: {detalles}")


def seleccionar_modo():
    print("Modo de trabajo:")
    print(" 1. Extraer subtitulos")
    print(" 2. Muxear subtitulos traducidos")
    print(" 3. Extraer y luego muxear")
    raw = input("Elige una opcion [1-3] (default 3): ").strip()
    if raw in {'1', '2', '3'}:
        return int(raw)
    return 3


def seleccionar_idiomas(idiomas):
    print("Idiomas encontrados:")
    for i, lang in enumerate(idiomas, 1):
        print(f" {i}. {lang}")
    sel = input("Elige los numeros de los idiomas a extraer (ej. 1,3): ")
    nums = [int(x.strip()) for x in sel.split(',') if x.strip().isdigit()]
    return [idiomas[i - 1] for i in nums if 0 < i <= len(idiomas)]


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
        print(f"[WARN] No se pudo leer subtitulos existentes de '{os.path.basename(video_path)}': {err.strip()}")
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

    print(f"\nNo se pudo asociar automatico: {shorten_name(sub_name)}")
    print("Selecciona el video destino:")
    for i, v in enumerate(videos, 1):
        print(f" {i}. {shorten_name(v)}")
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
        print("\n[WARN] Subtitulos traducidos sin video base detectado:")
        for s in sin_video:
            print(f"  - {shorten_name(s)}")

    if not por_video:
        print("\nNo se detectaron subtitulos traducidos para muxear.")
        print("Tip: se esperan nombres tipo *_subX_es-419.srt o *_eng_es-419.ass")
        return []

    muxeados = []
    ok, fail = 0, 0
    print("\n[RUN] Iniciando mux de subtitulos traducidos...\n")

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
            print(
                f"[WARN] '{video_name}' no es MKV; se crea salida separada ({out_name})."
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
                    print(
                        f"[ERR] Mux generado pero no se pudo reemplazar "
                        f"'{shorten_name(video_name)}'. Detalle: {exc}"
                    )
                    continue
                final_label = video_name
                if backup_path:
                    print(f"[OK] Backup original: {os.path.basename(backup_path)}")

            ok += 1
            muxeados.extend(os.path.join(path, s) for s in sub_files)
            print(
                f"[OK] Mux OK: {shorten_name(video_name)} + {len(sub_files)} "
                f"subtitulo(s) -> {shorten_name(final_label)}"
            )
        else:
            fail += 1
            print(
                f"[ERR] Error al muxear '{shorten_name(video_name)}' ({mux_tool_name}). "
                f"Detalle: {_last_error_line(err)}"
            )

    print("\n-------- Resumen Mux --------")
    print(f"Videos muxeados correctamente: {ok}")
    if fail:
        print(f"Videos con error en mux: {fail}")

    return sorted(set(muxeados))


def preguntar_borrado_subs(sub_paths, auto_delete=False, auto_delete_originales=False):
    if not sub_paths:
        return

    print("\nSubtitulos traducidos usados en mux:")
    for sub_path in sub_paths:
        print(f"  - {shorten_name(os.path.basename(sub_path))}")

    if auto_delete:
        print("\n[AUTO] Eliminacion de subtitulos muxeados: SI")
    elif not ask_yes_no("\nQuieres eliminar estos subtitulos ya muxeados?", default=False):
        print("Se conservaron los subtitulos.")
        return

    originales_relacionados = detectar_subs_originales_relacionados(sub_paths)
    a_borrar = list(sub_paths)
    eliminar_originales = False
    if originales_relacionados:
        print("\nSubtitulos originales relacionados detectados:")
        for sub_path in originales_relacionados:
            print(f"  - {shorten_name(os.path.basename(sub_path))}")
        if auto_delete_originales:
            eliminar_originales = True
            print("[AUTO] Eliminacion de subtitulos originales relacionados: SI")
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
            print(f"[WARN] No se pudo borrar '{os.path.basename(sub_path)}': {exc}")

    print(f"Subtitulos borrados: {borrados}")
    if originales_relacionados and not eliminar_originales:
        print("Se conservaron los subtitulos originales relacionados.")
    if errores:
        print(f"[WARN] Errores al borrar: {errores}")


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
        '--si',
        action='store_true',
        help=(
            "Modo semi-automatico para mux: reemplaza original, sin backup, "
            "borra subtitulos muxeados y originales relacionados sin preguntar."
        ),
    )
    args = parser.parse_args()

    base_dir = os.path.abspath(os.path.dirname(__file__))
    bulk_dir = os.path.join(base_dir, 'SUBS_BULK')
    os.makedirs(bulk_dir, exist_ok=True)

    print(f"Carpeta de trabajo: {bulk_dir}")
    modo = seleccionar_modo()

    if modo in {1, 3}:
        idiomas, streams = scan_directory(bulk_dir)
        if not streams:
            print("\nNo se detectaron pistas de subtitulos en videos dentro de SUBS_BULK.")
        else:
            idiomas_seleccionados = seleccionar_idiomas(idiomas)
            if not idiomas_seleccionados:
                print("No se selecciono ningun idioma para extraer.")
            else:
                print(f"\n[RUN] Iniciando extraccion para: {', '.join(idiomas_seleccionados)}\n")
                extract_subs(bulk_dir, streams, idiomas_seleccionados)

    muxed_sub_paths = []
    if modo in {2, 3}:
        if args.si:
            reemplazar = True
            backup = False
            print("\n[AUTO --si] Reemplazar original: SI")
            print("[AUTO --si] Crear backup: NO")
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
