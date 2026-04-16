"""
chunker.py
Splits cleaned JSON documents into overlapping text chunks
with rich metadata for retrieval.
"""

import json
import logging
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)

RAW_DIR    = Path("data/raw")
CHUNKS_DIR = Path("data/chunks")

# Chunk config — tuned for RAG over academic/dept content
CHUNK_SIZE    = 512   # characters
CHUNK_OVERLAP = 100

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def load_documents(raw_dir: Path = RAW_DIR) -> Iterator[dict]:
    """Yield cleaned JSON docs, skipping index files."""
    for path in sorted(raw_dir.glob("*.json")):
        if path.name.startswith("_"):
            continue
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
            if doc.get("text", "").strip():
                yield doc
        except Exception as e:
            log.warning("Could not load %s: %s", path.name, e)


def chunk_document(doc: dict) -> list[dict]:
    """Split a single document into chunks with metadata."""
    text      = doc["text"]
    url       = doc.get("url", "")
    title     = doc.get("title", "")
    page_type = doc.get("page_type", "general")
    courses   = doc.get("course_codes", [])

    splits = splitter.split_text(text)
    chunks = []
    for i, chunk_text in enumerate(splits):
        if not chunk_text.strip():
            continue
        chunks.append({
            "text":         chunk_text,
            "url":          url,
            "title":        title,
            "page_type":    page_type,
            "course_codes": courses,
            "chunk_index":  i,
            "chunk_total":  len(splits),
        })
    return chunks


def run_chunking(
    raw_dir:    Path = RAW_DIR,
    chunks_dir: Path = CHUNKS_DIR,
) -> list[dict]:
    chunks_dir.mkdir(parents=True, exist_ok=True)

    all_chunks = []
    doc_count  = 0

    for doc in load_documents(raw_dir):
        doc_chunks = chunk_document(doc)
        all_chunks.extend(doc_chunks)
        doc_count += 1

    # Save all chunks to a single file for inspection
    out_path = chunks_dir / "all_chunks.json"
    out_path.write_text(
        json.dumps(all_chunks, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    log.info(
        "Chunking complete. Docs: %d | Chunks: %d | Saved: %s",
        doc_count, len(all_chunks), out_path
    )
    return all_chunks
