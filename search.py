"""Day 2, step 2: ask a question, see which chunks the search finds.

Run:  python search.py
No AI answer yet - this only shows the retrieved chunks, so you can judge
whether the search is finding the right pieces. (The AI answer comes next.)
"""
import chromadb
from sentence_transformers import SentenceTransformer

DB_DIR = "chroma_db"
COLLECTION = "docs"
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 5


def search(question, model, collection, k=TOP_K):
    q_emb = model.encode([question], normalize_embeddings=True).tolist()
    res = collection.query(query_embeddings=q_emb, n_results=k)
    hits = []
    for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
        hits.append({"text": doc, "score": 1 - dist, **meta})
    return hits


def main():
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)

    print("Type a question (or 'q' to quit).")
    while True:
        question = input("\nQuestion> ").strip()
        if question.lower() in {"q", "quit", "exit"}:
            break
        if not question:
            continue
        for i, h in enumerate(search(question, model, collection), start=1):
            print(f"\n[{i}] score={h['score']:.2f}  {h['source']}  >  {h['section']}")
            print(h["text"][:300].replace("\n", " ") + " ...")


if __name__ == "__main__":
    main()
