# src/retrieval.py

import json
from pathlib import Path
from typing import List, Dict

import numpy as np
import faiss

ROOT = Path(__file__).resolve().parent.parent
INDEX_DIR = ROOT / "index"

EMBEDDINGS_PATH = INDEX_DIR / "embeddings.npy"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
LOOKUP_PATH = INDEX_DIR / "lookup.json"


def load_index():
    embeddings = np.load(EMBEDDINGS_PATH)
    index = faiss.read_index(str(FAISS_INDEX_PATH))

    with LOOKUP_PATH.open("r", encoding="utf-8") as f:
        lookup = json.load(f)

    return embeddings, index, lookup

def retrieve(query_embedding: np.ndarray, index, lookup, k: int = 5) -> List[Dict]:
    scores, ids = index.search(query_embedding, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        entry = lookup[str(idx)]
        entry["score"] = float(score)
        results.append(entry)

    return results
