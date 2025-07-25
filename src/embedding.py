# src/embedding.py

import torch
from typing import List
from transformers import (
    DPRContextEncoder,
    DPRContextEncoderTokenizer,
    DPRQuestionEncoder,
    DPRQuestionEncoderTokenizer
)


def load_dpr_models(
    ctx_model_name: str = 'facebook/dpr-ctx_encoder-single-nq-base', 
    q_model_name: str = 'facebook/dpr-question_encoder-single-nq-base'
    ):
    """
    Carga correctamente los tokenizers y encoders de contexto y pregunta.

    Returns:
        ctx_tokenizer, ctx_encoder, q_tokenizer, q_encoder
    """
    # 1. Contexto
    ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(ctx_model_name)
    ctx_encoder   = DPRContextEncoder.from_pretrained(ctx_model_name)

    # 2. Pregunta
    q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(q_model_name)
    q_encoder   = DPRQuestionEncoder.from_pretrained(q_model_name)

    return ctx_tokenizer, ctx_encoder, q_tokenizer, q_encoder


def encode_chunks(
    chunks: List[str],
    tokenizer,
    encoder,
    batch_size: int,
    max_length: int
    ):
    """
    Codifica una lista de tokens en batches para generar embeddings,
    usando GPU si esta disponible, sino CPU.
    """
    # Detectar el device en cada llamada
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder.to(device)
    emb_list = []

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors = "pt",
            padding = True,
            truncation = True,
            max_length = max_length
        ).to(device)

        with torch.no_grad():
            outputs = encoder(**inputs)
        emb_list.append(outputs.pooler_output)

    return torch.cat(emb_list, dim = 0)

if __name__ == "__main__":
    # Prueba rápida de carga
    ctx_tok, ctx_enc, q_tok, q_enc = load_dpr_models()
    print("✅ Modelos DPR cargados correctamente")