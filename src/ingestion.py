# src/ingestion.py

import os
import wget
import fitz
import re
import json
from typing import Optional, Dict, List, Tuple
from collections import Counter



def download_file(url: str, filename: str):
    """
    Funcion dedicada a descargar el archivo desde una URL especifica.

    Args:
        url: direccion del recurso.
        filename: ruta donde guardar el archivo.
    """
    if not os.path.isfile(filename):
        wget.download(url, out=filename)
        print(f"Archivo descargado: {filename}")
    else:
        print(f"El archivo {filename} ya existe, no se descargara")


def chunk_text(
    text: str,
    tokenizer,
    chunk_size: int,
    overlap: int
    ):
    """
    Divide un texto en fragmentos (chunks) basados en tokens, con solapamiento.

    Args:
        text: cadena completa a fragmentar.
        tokenizer: instancia de PreTrainedTokenizer de HF.
        chunk_size: número de tokens por fragmento.
        overlap: tokens que se repiten entre fragmentos.

    Returns:
        Lista de fragmentos de texto.
    """
    token_ids = tokenizer.encode(text)
    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(token_ids), step):
        chunk_ids = token_ids[i : i + chunk_size]
        chunks.append(tokenizer.decode(chunk_ids))
        if i + chunk_size >= len(token_ids):
            break
    return chunks


def read_and_chunk_text(
    filename: str,
    tokenizer,
    chunk_size: int,
    overlap: int
    ):
    """
    Lee un archivo y lo divide en chunks usando tokenización.

    Args:
        filename: ruta al archivo de texto.
        tokenizer: instancia de PreTrainedTokenizer.
        chunk_size: número de tokens por chunk.
        overlap: tokens superpuestos entre chunks.

    Returns:
        Lista de chunks de texto.
    """
    with open(filename, 'r', encoding='utf-8') as f:
        text = f.read()
    return chunk_text(text, tokenizer, chunk_size, overlap)


def pdf_to_txt(
    pdf_path: str,
    txt_path: str,
    metadata_path: Optional[str] = None,
    line_sep: str = "\n",
    cleanup_hyphens: bool = True,
    normalize_whitespace: bool = True,
    remove_headers_footers: bool = True,
    hf_lines: int = 3
) -> dict:
    """
    Convierte un PDF a TXT para RAG, eliminando:
      - headers/footers repetidos
      - líneas de solo número de página (p.ej. '12', 'Page 3', '3 of 10')
      - líneas de solo numeración de lista (p.ej. '58.', '59.')
      - líneas de solo viñetas ('•' o '◦')
      - guiones partidos al final de línea
      - espacios/tab extra
    """

    # Patrones de línea a eliminar
    page_num_pattern = re.compile(r'^\s*(?:Page\s*)?\d+(?:\s+of\s+\d+)?\s*$', re.IGNORECASE)
    list_item_pattern = re.compile(r'^\s*(?:\d+\.\s*|[•◦]+\s*)$')

    # 1. Abrir documento y extraer metadatos
    doc = fitz.open(pdf_path)
    metadata = doc.metadata or {}

    # 2. Leer cada página
    pages_text: List[str] = [doc.load_page(i).get_text("text") for i in range(doc.page_count)]  # type: ignore

    # 3. Detectar headers/footers repetidos
    headers, footers = set(), set()
    if remove_headers_footers and doc.page_count > 1:
        top  = [p.splitlines()[:hf_lines] for p in pages_text]
        bot  = [p.splitlines()[-hf_lines:] for p in pages_text]
        cnt_top    = Counter(line.strip() for blk in top for line in blk if line.strip())
        cnt_bot    = Counter(line.strip() for blk in bot for line in blk if line.strip())
        thr = doc.page_count // 2
        headers = {l for l,c in cnt_top.items() if c > thr}
        footers = {l for l,c in cnt_bot.items() if c > thr}

    # 4. Limpiar página a página
    cleaned_pages: List[str] = []
    for raw in pages_text:
        lines = raw.splitlines()

        # a) eliminar headers/footers
        if remove_headers_footers:
            lines = [
                ln for ln in lines
                if ln.strip() not in headers and ln.strip() not in footers
            ]

        # b) eliminar líneas de número de página, lista y viñetas
        lines = [
            ln for ln in lines
            if not page_num_pattern.match(ln)
            and not list_item_pattern.match(ln)
        ]

        # c) reconstruir texto
        text = line_sep.join(lines)

        # d) unir guiones partidos al final de línea
        if cleanup_hyphens:
            text = re.sub(
                r'(\w+)-\s*\n\s*(\w+)',
                r'\1\2',
                text
            )

        # e) normalizar espacios/tab
        if normalize_whitespace:
            text = re.sub(r'[ \t]+', ' ', text)

        cleaned_pages.append(text)

    # 5. Unir todo en un único bloque
    final_text = line_sep.join(cleaned_pages)

    # 6. Guardar TXT
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(final_text)

    # 7. (Opcional) Guardar metadatos
    if metadata_path:
        with open(metadata_path, 'w', encoding='utf-8') as mf:
            json.dump(metadata, mf, ensure_ascii=False, indent=2)

    return metadata