#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extraer_subtitulos_multi.py
Versión 2.0 – 07-ago-2025

• Extrae todos los subtítulos de archivos .mkv en el directorio actual.
• Respeta el formato original (ASS → .ass | SubRip → .srt).
• Otros formatos se convierten a .srt.
"""

import subprocess
import os
import json
from collections import defaultdict

VIDEO_EXTS = {'.mkv', '.mp4', '.avi'}
FFPROBE_FIELDS = 'stream=index,codec_type,codec_name:stream_tags=language'


def _run(cmd):
    """Ejecuta un comando y devuelve (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.returncode, result.stdout.decode(), result.stderr.decode()


def scan_directory(path):
    """
    Escanea el directorio y devuelve:
      • idiomas_disponibles: lista ordenada de códigos ISO únicos
      • streams_info: [(archivo, index, iso, codec_name), ...]
    """
    idiomas, streams = set(), []

    for f in os.listdir(path):
        if os.path.splitext(f)[1].lower() not in VIDEO_EXTS:
            continue

        code, out, err = _run([
            'ffprobe', '-v', 'error',
            '-select_streams', 's',
            '-show_entries', FFPROBE_FIELDS,
            '-of', 'json', os.path.join(path, f)
        ])
        if code != 0:
            print(f"⚠️  ffprobe falló en '{f}': {err.strip()}")
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
            print(f"⚠️  No se pudo leer la salida de ffprobe para '{f}'.")

    return sorted(idiomas), streams


def extract_subs(path, streams, idiomas_sel):
    out_dir = os.path.join(path, 'SUBS_BULK')
    os.makedirs(out_dir, exist_ok=True)

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
            # Cualquier otro formato → convertir a SRT
            ext, codec_opt = '.srt', 'srt'

        out_file = f"{base}_{lang}_sub{idx}{ext}"
        out_path = os.path.join(out_dir, out_file)

        code, _, err = _run([
            'ffmpeg', '-y', '-i', os.path.join(path, file_name),
            '-map', f'0:{idx}', '-c:s', codec_opt, out_path
        ])

        if code == 0 and os.path.isfile(out_path):
            ok += 1
            print(f"✅ Extraído {lang} | {codec.upper():<7} → {out_file}")
        else:
            fail += 1
            errores[file_name].append((idx, lang, codec))
            print(f"❌ Error al extraer {lang} | {codec.upper()} de '{file_name}'.\n   ffmpeg: {err.strip().splitlines()[-1]}")

    # Resumen
    print("\n──────── Resumen ────────")
    print(f"Subtítulos extraídos correctamente: {ok}")
    if fail:
        print(f"Subtítulos con error: {fail}")
        for f, lst in errores.items():
            detalles = ', '.join(f"{i}:{l}" for i, l, _ in lst)
            print(f"  • {f}: {detalles}")


def seleccionar_idiomas(idiomas):
    print("Idiomas encontrados:")
    for i, lang in enumerate(idiomas, 1):
        print(f" {i}. {lang}")
    sel = input("Elige los números de los idiomas a extraer (ej. 1,3): ")
    nums = [int(x.strip()) for x in sel.split(',') if x.strip().isdigit()]
    return [idiomas[i - 1] for i in nums if 0 < i <= len(idiomas)]


if __name__ == '__main__':
    DIR = os.path.abspath(os.path.dirname(__file__))
    idiomas, streams = scan_directory(DIR)

    if not idiomas:
        print("No se encontraron subtítulos en los archivos .mkv del directorio.")
        exit()

    idiomas_seleccionados = seleccionar_idiomas(idiomas)
    if not idiomas_seleccionados:
        print("No se seleccionó ningún idioma. Saliendo.")
        exit()

    print(f"\n▶️  Iniciando extracción para: {', '.join(idiomas_seleccionados)}\n")
    extract_subs(DIR, streams, idiomas_seleccionados)
