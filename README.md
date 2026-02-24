# SubLLM

Herramientas para trabajar subtitulos locales:

- Extraer subtitulos desde videos (`.mkv`, `.mp4`, `.avi`).
- Traducir subtitulos `.srt` y `.ass` usando un modelo local de Ollama.
- Mantener etiquetas/timing y generar archivos listos para usar.

## Scripts incluidos

- `traducir_subtitulos.py`: traduce subtitulos con contexto (resumen + guia de tono).
- `gestionar_subtitulos.py`: extrae y muxea subtitulos en `SUBS_BULK/`.

## Requisitos

- Python 3.10 o superior.
- Ollama instalado y activo (para traduccion).
- Un modelo instalado en Ollama (ejemplo: `gemma3:4b`).
- `ffmpeg` y `ffprobe` en PATH (para extraccion).
- Recomendado: MKVToolNix (`mkvmerge`/`mkvextract`) para escaneo/extraccion/mux mas rapido y robusto en MKV.
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
python traducir_subtitulos.py --interactive
```

Modo directo:

```bash
python traducir_subtitulos.py --in "archivo.ass" --target Spanish --model gemma3:4b
```

Nota:

- Si pasas solo nombre de archivo (sin ruta), `traducir_subtitulos.py` busca el input en `SUBS_BULK/`.
- La salida por defecto tambien se guarda en `SUBS_BULK/`.

Opciones utiles:

- `--out`: ruta de salida.
- `--batch`: traduce en lote todos los `.ass/.srt` de `SUBS_BULK/`.
- `--parallel-files N`: cantidad de archivos en paralelo en `--batch` o multi-seleccion (usa subprocesos).
- `--num-threads N`: hilos para Ollama (`6` por defecto).
- `--overwrite`: con `--batch`, sobrescribe salidas existentes.
- En `--batch` y multi-seleccion ahora se imprime progreso global de archivos: `ok/skipped/failed/remaining`.
- En modo paralelo (`--parallel-files > 1`) se muestra barra de progreso por archivos completados.
- `--batch-size`: tamano de lote para traduccion.
- `--ass-mode line|segment`: modo de traduccion en archivos ASS.
- `--skip-summary`: omite resumen/contexto para mayor velocidad.
- `--limit N`: traduce solo los primeros N bloques.
- `--fast`: aplica perfil rapido (prioriza velocidad sobre cobertura).
- `--one-shot`: intenta traducir en un lote grande cuando el contexto lo permite.
- `--bench`: imprime metricas detalladas de rendimiento por llamada.

## Rendimiento y reintentos (actualizado)

El flujo de traduccion fue optimizado para reducir llamadas extra al modelo y mejorar estabilidad en equipos con recursos limitados:

- Mejor manejo de respuestas JSON parciales para evitar reintentos costosos.
- Control de profundidad de split de lotes para evitar explosiones de llamadas.
- Presupuesto de reintentos en modo rapido (`--fast`) para priorizar velocidad.
- Ajuste adaptativo nativo del tamano de lote cuando detecta splits/fallos altos, para estabilizar tiempos sin flags extra.
- Resumen de metricas por archivo tambien en modo paralelo (`--parallel-files > 1`).

En el resumen final veras, entre otros:

- `Retry counters`: reintentos por tipo.
- `Fast retry budget`: items recortados por presupuesto en `--fast`.
- `Split stats`: recursiones/cortes de split.
- `Top flagged reasons`: razones mas comunes de lineas marcadas.
- `Ollama calls`: numero total de llamadas al modelo.

## Recomendacion para GPU limitada

Si Ollama usa una sola GPU con memoria limitada:

- Usa modo normal por defecto para mejor cobertura de traduccion.
- Activa `--fast --one-shot` solo cuando necesites maxima velocidad.
- `--parallel-files` usa `3` por defecto.
- Si notas contencion en otros modelos/equipos, prueba `--parallel-files 2`.

Ejemplo de lote recomendado:

```bash
python traducir_subtitulos.py --batch --in "*S01E0[345]*fre_sub2.ass" --target Spanish --fast --one-shot
```

Salida por defecto:

- Si el destino es espanol: `*_es-419.srt` o `*_es-419.ass`.
- Si el destino es ingles: `*_en.srt` o `*_en.ass`.

## Uso: extraccion y mux de subtitulos

Con videos y subtitulos dentro de `SUBS_BULK/`:

```bash
python gestionar_subtitulos.py
```

Modo semi-automatico para mux (sin preguntas de confirmacion extra en el flujo de mux):

```bash
python gestionar_subtitulos.py --si
```

El script:

- Detecta pistas de subtitulo disponibles por idioma.
- Permite extraer, muxear o hacer ambas cosas.
- Conserva formato (`ass`/`srt`) o convierte a `srt` cuando aplica.
- En `.mkv`, si MKVToolNix esta disponible, usa `mkvmerge -J` para escaneo y `mkvextract` para extraer (mas rapido).
- Omite automaticamente subtitulos basados en imagen (PGS/VobSub/DVB) porque requieren OCR.
- Al muxear, agrega metadatos de idioma/titulo a subtitulos externos.
- Si hay subtitulo en espanol, lo marca como `default`.
- Si `mkvmerge` esta disponible, lo usa automaticamente para mux de `.mkv`.
- Puede reemplazar el video original (con backup opcional).
- Al final pregunta si deseas borrar subtitulos ya muxeados y, opcionalmente, los originales relacionados.
- Con `--si` (si eliges mux), aplica automaticamente: reemplazar original, sin backup, borrar subtitulos muxeados y originales relacionados.

## Flujo recomendado

1. Ejecuta `gestionar_subtitulos.py` para sacar subtitulos del video.
2. Revisa el archivo extraido (`.ass` o `.srt`).
3. Traduce con `traducir_subtitulos.py`.
4. Prueba el resultado en tu reproductor y ajusta parametros si hace falta.

## Notas de sincronizacion del repo

Este repositorio usa una politica de lista blanca en `.gitignore`: por defecto se ignora todo y solo se versionan los archivos del proyecto. Esto evita sincronizar videos, subtitulos temporales y archivos externos por error.
