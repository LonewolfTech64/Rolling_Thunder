from pathlib import Path
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

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

def build_corpus_with_metadata():
    corpus = load_corpus()
    metadata = load_metadata()

    combined = []
    for entry in corpus:
        doc_id = entry["doc_id"]
        text = entry["text"]

        meta = metadata.get(doc_id, {})
        combined.append({
            "id": entry["id"],
            "doc_id": doc_id,
            "text": text,
            "metadata": meta
        })

    return combined

if __name__ == "__main__":
    combined = build_corpus_with_metadata()
    print(f"Loaded {len(combined)} documents with metadata.")
