# src/database.py

import numpy as np
import faiss

def build_faiss_index_cpu(
    embeddings: np.ndarray,
):
    """
    Crea y devuelve un índice FAISS usando L2 para los embeddings.
    """
    emb = np.ascontiguousarray(embeddings.astype('float32'))
    dim = emb.shape[1]

    index = faiss.IndexFlatL2(dim)

    index.add(emb) # type: ignore
    return index