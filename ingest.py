"""Day 1: load documents from ./data into one uniform corpus (corpus.jsonl).

Every record has: id, source, section, date, text.
Those metadata fields are what we filter on later (source / date / section).

Run:  python ingest.py
"""
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader

DATA_DIR = Path("data")
OUT_FILE = Path("corpus.jsonl")


def load_markdown(path):
    """Split a markdown file into sections at #, ##, ### headings."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    parts = re.split(r"(?m)^(#{1,3} .+)$", text)
    sections = []
    if parts[0].strip():
        sections.append((path.stem, parts[0]))
    for i in range(1, len(parts), 2):
        heading = parts[i].lstrip("# ").strip()
        body = parts[i + 1] if i + 1 < len(parts) else ""
        if body.strip():
            sections.append((heading, body))
    return sections


def load_pdf(path):
    """One section per page (page number is useful for citations)."""
    reader = PdfReader(str(path))
    sections = []
    for n, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            sections.append((f"page {n}", text))
    return sections


def load_html(path):
    html = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer"]):
        tag.decompose()
    title = soup.title.string.strip() if soup.title and soup.title.string else path.stem
    text = soup.get_text(separator="\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return [(title, text)] if text.strip() else []


def load_docx(path):
    doc = Document(str(path))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(path.stem, text)] if text.strip() else []


LOADERS = {
    ".md": load_markdown,
    ".markdown": load_markdown,
    ".pdf": load_pdf,
    ".html": load_html,
    ".htm": load_html,
    ".docx": load_docx,
}


def main():
    if not DATA_DIR.exists():
        raise SystemExit(f"Create a '{DATA_DIR}/' folder and put your documents in it.")

    records, skipped, files = [], [], 0
    for path in sorted(DATA_DIR.rglob("*")):
        if not path.is_file():
            continue
        loader = LOADERS.get(path.suffix.lower())
        if loader is None:
            skipped.append(path.name)
            continue
        try:
            sections = loader(path)
        except Exception as e:  # keep going; report at the end
            skipped.append(f"{path.name} ({e})")
            continue
        files += 1
        date = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).date().isoformat()
        rel = str(path.relative_to(DATA_DIR))
        for idx, (section, text) in enumerate(sections):
            records.append(
                {
                    "id": f"{rel}::{idx}",
                    "source": rel,
                    "section": section,
                    "date": date,
                    "text": text.strip(),
                }
            )

    with OUT_FILE.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    total_chars = sum(len(r["text"]) for r in records)
    print(f"Files loaded : {files}")
    print(f"Sections     : {len(records)}")
    print(f"Total chars  : {total_chars:,} (~{total_chars // 4:,} tokens)")
    print(f"Skipped      : {len(skipped)}")
    for s in skipped[:10]:
        print("   -", s)
    print(f"Wrote {OUT_FILE}")


if __name__ == "__main__":
    main()
