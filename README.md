# SubLLM

Herramientas para trabajar subtitulos locales:

- Extraer subtitulos desde videos (`.mkv`, `.mp4`, `.avi`).
- Traducir subtitulos `.srt` y `.ass` usando un modelo local de Ollama.
- Mantener etiquetas/timing y generar archivos listos para usar.

## Scripts incluidos

- `translate_subs.py`: traduce subtitulos con contexto (resumen + guia de tono).
- `EXTRAER_subs_v1.py`: extrae pistas de subtitulos de videos con `ffprobe`/`ffmpeg`.

## Requisitos

- Python 3.10 o superior.
- Ollama instalado y activo (para traduccion).
- Un modelo instalado en Ollama (ejemplo: `gemma3:4b`).
- `ffmpeg` y `ffprobe` en PATH (para extraccion).
- Opcional: `rich` para barras de progreso mas claras.

## Instalacion rapida

```bash
python -m pip install --upgrade pip
python -m pip install rich
```

Verificar Ollama:

```bash
ollama list
```

## Uso: traduccion de subtitulos

Modo interactivo (recomendado):

```bash
python translate_subs.py --interactive
```

Modo directo:

```bash
python translate_subs.py --in "archivo.ass" --target Spanish --model gemma3:4b --fast
```

Nota:

- Si pasas solo nombre de archivo (sin ruta), `translate_subs.py` busca el input en `SUBS_BULK/`.
- La salida por defecto tambien se guarda en `SUBS_BULK/`.

Opciones utiles:

- `--out`: ruta de salida.
- `--batch-size`: tamano de lote para traduccion.
- `--ass-mode line|segment`: modo de traduccion en archivos ASS.
- `--skip-summary`: omite resumen/contexto para mayor velocidad.
- `--limit N`: traduce solo los primeros N bloques.
- `--fast`: aplica perfil rapido.

Salida por defecto:

- Si el destino es espanol: `*_es-419.srt` o `*_es-419.ass`.
- Si el destino es ingles: `*_en.srt` o `*_en.ass`.

## Uso: extraccion de subtitulos desde video

En la carpeta con videos:

```bash
python EXTRAER_subs_v1.py
```

El script:

- Detecta pistas de subtitulo disponibles por idioma.
- Te pide seleccionar idiomas.
- Extrae resultados a `SUBS_BULK/`.
- Conserva formato (`ass`/`srt`) o convierte a `srt` cuando aplica.

## Flujo recomendado

1. Ejecuta `EXTRAER_subs_v1.py` para sacar subtitulos del video.
2. Revisa el archivo extraido (`.ass` o `.srt`).
3. Traduce con `translate_subs.py`.
4. Prueba el resultado en tu reproductor y ajusta parametros si hace falta.

## Notas de sincronizacion del repo

Este repositorio usa una politica de lista blanca en `.gitignore`: por defecto se ignora todo y solo se versionan los archivos del proyecto. Esto evita sincronizar videos, subtitulos temporales y archivos externos por error.
