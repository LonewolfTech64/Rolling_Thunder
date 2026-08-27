import os
import json
import textwrap

ALL_DIR = r"C:\von_Clausewitz_Paper\ALL"
METADATA_PATH = r"C:\Users\dalea\PycharmProjects\PythonProject1\data\metadata.json"
OUTPUT_PATH = r"C:\Users\dalea\PycharmProjects\PythonProject1\data\corpus.jsonl"

CHUNK_SIZE = 700  # characters


def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def read_raw_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().strip()


def chunk_text(text, size=CHUNK_SIZE):
    return textwrap.wrap(text, size)


def extract_doc_id(filename):
    # Remove extension
    base = os.path.splitext(filename)[0]
    return base.lower()  # matches metadata.json keys


def rebuild():
    metadata = load_metadata()

    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for filename in os.listdir(ALL_DIR):
            if not filename.endswith(".txt"):
                continue

            full_path = os.path.join(ALL_DIR, filename)
            doc_id = extract_doc_id(filename)

            text = read_raw_text(full_path)

            # Get metadata entry
            meta = metadata.get(doc_id, {})

            # If missing bias, add placeholder
            if "bias" not in meta:
                meta["bias"] = ["to_be_classified"]

            # Build base object
            base_obj = {
                "id": doc_id,
                "source": meta.get("source", "unknown"),
                "type": meta.get("type", "unknown"),
                "date": meta.get("date", "unknown"),
                "tone": meta.get("tone", "unknown"),
                "bias": meta.get("bias", ["to_be_classified"]),
            }

            # Chunk text
            chunks = chunk_text(text)

            for i, chunk in enumerate(chunks):
                obj = base_obj.copy()
                obj["chunk"] = i
                obj["text"] = chunk

                out.write(json.dumps(obj) + "\n")

    print("Corpus rebuilt successfully:", OUTPUT_PATH)


if __name__ == "__main__":
    rebuild()

