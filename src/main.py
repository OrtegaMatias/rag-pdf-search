# src/main.py

import random
import os
import torch
import faiss
import gradio as gr
import time
from ingestion import (
    pdf_to_txt,
    read_and_chunk_text
)
from embedding import (
    load_dpr_models,
    encode_chunks
)
from database import build_faiss_index_cpu
from retriever import search_relevant_contexts

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def load_paragraphs(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        paragraphs = [line.strip() for line in f.readlines()]
    return paragraphs

def preparar_sistema():
    embeddings_path = "embeddings.pt"
    faiss_index_path = "faiss.index"
    paragraphs_path = "paragraphs.txt"

    # 1. Convertir PDF si no existe TXT
    metadata = pdf_to_txt(
        pdf_path="Mancha.pdf",
        txt_path="Ficha.txt",
        metadata_path=None,
        line_sep="\n",
        cleanup_hyphens=True,
        normalize_whitespace=True,
        remove_headers_footers=True,
        hf_lines=3
    )
    print("📘 Metadatos extraídos:", metadata)

    # 2. Cargar modelos
    ctx_tokenizer, ctx_encoder, q_tokenizer, q_encoder = load_dpr_models()

    # 3. Cargar o crear chunks
    if os.path.exists(paragraphs_path):
        paragraphs = load_paragraphs(paragraphs_path)
        print(f"✅ Se cargaron {len(paragraphs)} párrafos.")
    else:
        paragraphs = read_and_chunk_text(
            "Ficha.txt",
            ctx_tokenizer,
            chunk_size=200,
            overlap=40
        )
        with open(paragraphs_path, "w", encoding="utf-8") as f:
            for p in paragraphs:
                f.write(p.replace("\n", " ") + "\n")
        print(f"✅ Se guardaron {len(paragraphs)} párrafos.")

    # 4. Embeddings + FAISS
    index_time = 0
    if os.path.exists(embeddings_path) and os.path.exists(faiss_index_path):
        print("🧠 Cargando embeddings e índice...")
        embeddings = torch.load(embeddings_path)
        index = faiss.read_index(faiss_index_path)
    else:
        print("🧠 Generando embeddings e índice FAISS...")
        start = time.time()
        embeddings = encode_chunks(
            paragraphs,
            ctx_tokenizer,
            ctx_encoder,
            batch_size=32,
            max_length=256
        )
        torch.save(embeddings, embeddings_path)
        context_embeddings_np = embeddings.detach().cpu().numpy()
        index = build_faiss_index_cpu(context_embeddings_np)
        faiss.write_index(index, faiss_index_path)
        index_time = time.time() - start
        print(f"⏱️ Índice generado en {index_time:.2f} segundos")

    return paragraphs, q_tokenizer, q_encoder, index, index_time

# Cargar sistema una vez
paragraphs, q_tokenizer, q_encoder, index, index_time = preparar_sistema()

# Función para responder preguntas
def responder_pregunta(pregunta):
    start = time.time()
    D, I = search_relevant_contexts(
        pregunta,
        q_tokenizer,
        q_encoder,
        index,
        k=5,
        max_length=256
    )
    duration = time.time() - start

    resultados = []
    for rank, idx in enumerate(I[0], start=1):
        fragmento = paragraphs[idx]
        distancia = D[0][rank - 1]
        resultados.append(f"🔹 {rank}. {fragmento}\n📏 Distancia: {distancia:.2f}\n")

    footer = f"⏱️ Tiempo de recuperación: {duration:.2f} segundos"
    if index_time > 0:
        footer += f"\n⚙️ Tiempo de creación del índice: {index_time:.2f} segundos"
    return "\n".join(resultados) + "\n\n" + footer

def main():
    interfaz = gr.Interface(
        fn=responder_pregunta,
        inputs=gr.Textbox(lines=2, placeholder="Hazme una pregunta sobre el libro..."),
        outputs="text",
        title="📚 Pregunta sobre Don Quijote",
        description="Ingresa una pregunta y recupera los fragmentos más relevantes del texto."
    )
    interfaz.launch()

if __name__ == "__main__":
    main()