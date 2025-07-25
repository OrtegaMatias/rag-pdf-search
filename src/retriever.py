# src/retriever.py

import torch

def search_relevant_contexts(
    question: str,
    question_tokenizer,
    question_encoder,
    index,
    k: int,
    max_length: int
):
    """
    Busca los contextos más relevantes para una pregunta dada.
    """
    # tokenizar la pregunta
    inputs = question_tokenizer(
        question,
        return_tensors = "pt",
        padding = True,
        truncation = True,
        max_length = max_length
    )
    # Codificar para obtener embedding
    with torch.no_grad():
        outputs = question_encoder(**inputs)
    question_emb = outputs.pooler_output

    # Convertir a numpy y buscar
    question_np = question_emb.detach().cpu().numpy()
    D, I = index.search(question_np, k)
    return D, I