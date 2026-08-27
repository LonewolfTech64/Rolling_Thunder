from pathlib import Path
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
INDEX_DIR = ROOT / "index"

# Ensure index directory exists
INDEX_DIR.mkdir(exist_ok=True)

# Load combined corpus
CORPUS_PATH = DATA_DIR / "corpus.jsonl"
METADATA_PATH = DATA_DIR / "metadata.json"

def load_corpus():
    corpus = []
    with CORPUS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            corpus.append(json.loads(line))
    return corpus

def load_metadata():
    with METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)

def build_chunks():
    corpus = load_corpus()
    metadata = load_metadata()

    chunks = []
    for entry in corpus:
        # Safely extract doc_id
        doc_id = entry.get("doc_id", entry.get("id", "UNKNOWN_DOC"))

        text = entry.get("text", "")
        meta = metadata.get(doc_id, {})

        chunks.append({
            "id": entry.get("id", doc_id),
            "doc_id": doc_id,
            "text": text,
            "metadata": meta
        })

    return chunks


if __name__ == "__main__":
    print("Loading chunks...")
    chunks = build_chunks()
    print(f"Loaded {len(chunks)} chunks")

    model = SentenceTransformer("BAAI/bge-large-en")
    texts = [c["text"] for c in chunks]

    print("Building embeddings...")
    embeddings = model.encode(texts, convert_to_numpy=True)

    # Save embeddings
    np.save(INDEX_DIR / "embeddings.npy", embeddings)

    # Build FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # Build lookup table
    lookup = {i: {"id": c["id"], "doc_id": c["doc_id"], "metadata": c["metadata"]}
              for i, c in enumerate(chunks)}

    with (INDEX_DIR / "lookup.json").open("w", encoding="utf-8") as f:
        json.dump(lookup, f, indent=2)

    print(f"Done. Saved FAISS index to: {INDEX_DIR}")
