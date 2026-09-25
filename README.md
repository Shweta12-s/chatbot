# FastAPI Docs RAG Chatbot

A chatbot that answers questions from the FastAPI documentation.
Built step by step, with every improvement measured on a test set.

## Status

- [x] Ingestion (Markdown, PDF, HTML, DOCX)
- [x] Chunking, embeddings and vector search (Chroma)
- [x] Evaluation on 30 test questions (baseline)
- [ ] Chunk size experiments
- [ ] Hybrid search (keyword + vector)
- [ ] Re-ranking
- [ ] Answers with citations, refusal when the answer is not in the docs
- [ ] Caching and cost tracking
- [ ] FastAPI backend, UI, Docker, deployment

## Data

FastAPI documentation (155 text files, 1,832 sections, ~345,000 tokens).

## How to run

```
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -r requirements-day2.txt
.venv\Scripts\python.exe ingest.py        # reads documents -> corpus.jsonl
.venv\Scripts\python.exe build_index.py   # chunks + embeds -> chroma_db/
.venv\Scripts\python.exe search.py        # ask questions, see top 5 results
.venv\Scripts\python.exe eval.py          # score on eval_questions.json
```

## How the evaluation works

`eval_questions.json` has 30 questions. Each question lists the file that
contains the answer. For each question we check whether that file appears
in the top 5 search results.

- Hit rate @ 5: the right file is anywhere in the top 5
- Right file first: the right file is result number 1

## Results

| Version                                               | Hit rate @5 | Right file first |
| ----------------------------------------------------- | ----------- | ---------------- |
| Baseline: chunk 1800, overlap 200, vector search only | 87% (26/30) | 53% (16/30)      |

## Baseline failure analysis

1. "Change the title/description of the docs page" -> expected metadata.md.
   Search returned docs-UI files (it matched "docs page").
2. "Path parameter greater than 0" -> expected path-params-numeric-validations.md.
   Search returned generic path files (the file uses gt/ge, not "greater than").
3. "Share common logic between endpoints" -> expected dependencies/index.md.
   Search returned bigger-applications.md (the docs use the term "dependencies").
4. "Login form with username and password" -> expected request-forms.md.
   Ambiguous question: the security files are also plausible answers.

## Lessons so far

- My first 12 test questions were too easy (100% hit rate) because they copied
  file names. Harder questions written in everyday words gave a more honest
  baseline.
- Meaning-based search can confuse similar words (e.g. "path" in path
  parameters vs the PATH environment variable).
