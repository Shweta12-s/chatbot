"""Day 2, step 1: cut documents into chunks and store them in a searchable index.

Run:  python build_index.py

Three things happen here:
  1. CHUNK   - each section is cut into small overlapping pieces
  2. EMBED   - each piece is turned into a list of numbers (its "meaning")
  3. STORE   - the pieces + numbers are saved in a local database (chroma_db/)

CHUNK_SIZE and OVERLAP are the two numbers you will experiment with on days 4-5.
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

CORPUS_FILE = Path("corpus.jsonl")
DB_DIR = "chroma_db"
COLLECTION = "docs"

CHUNK_SIZE = 1800   # characters (about 450 tokens)
OVERLAP = 200       # characters shared between neighbouring chunks
MIN_CHARS = 50      # ignore tiny pieces
BATCH = 256

MODEL_NAME = "all-MiniLM-L6-v2"  # small, free, runs on your laptop


def chunk_text(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    records = [json.loads(line) for line in CORPUS_FILE.open(encoding="utf-8")]

    ids, texts, metas = [], [], []
    for r in records:
        for n, piece in enumerate(chunk_text(r["text"], CHUNK_SIZE, OVERLAP)):
            if len(piece.strip()) < MIN_CHARS:
                continue
            ids.append(f"{r['id']}::c{n}")
            texts.append(piece)
            metas.append({"source": r["source"], "section": r["section"], "date": r["date"]})
    print(f"Made {len(texts)} chunks from {len(records)} sections")

    print("Loading embedding model (first run downloads it)...")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)  # start fresh every run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    for i in range(0, len(texts), BATCH):
        batch_texts = texts[i : i + BATCH]
        embeddings = model.encode(batch_texts, normalize_embeddings=True).tolist()
        collection.add(
            ids=ids[i : i + BATCH],
            documents=batch_texts,
            embeddings=embeddings,
            metadatas=metas[i : i + BATCH],
        )
        print(f"  stored {min(i + BATCH, len(texts))}/{len(texts)}")

    print("Done. Now run: python search.py")


if __name__ == "__main__":
    main()
