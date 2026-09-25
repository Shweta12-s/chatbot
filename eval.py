"""Day 3: measure how good the search is.

For every question in eval_questions.json we run the search and check:
is the expected file among the top 5 results?

Run:  python eval.py
"""
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from search import COLLECTION, DB_DIR, MODEL_NAME, search

K = 5


def main():
    questions = json.loads(Path("eval_questions.json").read_text(encoding="utf-8"))
    model = SentenceTransformer(MODEL_NAME)
    collection = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION)

    hits, top1 = 0, 0
    for q in questions:
        results = search(q["question"], model, collection, k=K)
        sources = [r["source"].replace("\\", "/") for r in results]
        rank = sources.index(q["expected_file"]) + 1 if q["expected_file"] in sources else None

        if rank:
            hits += 1
            if rank == 1:
                top1 += 1
            print(f"HIT  (rank {rank})  {q['question']}")
        else:
            print(f"MISS            {q['question']}")
            print(f"      expected: {q['expected_file']}")
            print(f"      got     : {sources}")

    n = len(questions)
    print("\n===== RESULT =====")
    print(f"Questions        : {n}")
    print(f"Hit rate @ top {K}: {hits}/{n} = {hits / n:.0%}")
    print(f"Right file first : {top1}/{n} = {top1 / n:.0%}")
    print(f"(chunk settings and search type are in build_index.py / search.py)")


if __name__ == "__main__":
    main()
