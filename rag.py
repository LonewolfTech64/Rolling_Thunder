# src/rag.py

from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from retrieval import load_index, retrieve

MODEL_NAME = "BAAI/bge-large-en"

def embed_query(query: str) -> np.ndarray:
    model = SentenceTransformer(MODEL_NAME)
    model.to("cpu")
    emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return emb

def rag_answer(query: str, k: int = 5) -> Dict:
    embeddings, index, lookup = load_index()
    query_emb = embed_query(query)
    results = retrieve(query_emb, index, lookup, k=k)
    return {
        "query": query,
        "results": results
    }

if __name__ == "__main__":
    q = "Summarise JCS arguments for expanding Rolling Thunder and contrast them with CIA skepticism"
    out = rag_answer(q, k=10)
    print(out)
