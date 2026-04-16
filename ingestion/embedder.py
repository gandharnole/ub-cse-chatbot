"""
embedder.py
Generates embeddings for text chunks using nomic-embed-text via Ollama.
"""

import logging
import time
from typing import Optional

import ollama

log = logging.getLogger(__name__)

EMBED_MODEL = "nomic-embed-text"
BATCH_SIZE  = 32   # chunks per Ollama call
RETRY_LIMIT = 3
RETRY_DELAY = 2.0  # seconds


def embed_texts(texts: list[str]) -> Optional[list[list[float]]]:
    """
    Embed a batch of texts. Returns list of embedding vectors,
    or None on failure after retries.
    """
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            response = ollama.embed(model=EMBED_MODEL, input=texts)
            return response.embeddings
        except Exception as e:
            log.warning("Embed attempt %d/%d failed: %s", attempt, RETRY_LIMIT, e)
            if attempt < RETRY_LIMIT:
                time.sleep(RETRY_DELAY)
    return None


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Add an 'embedding' field to each chunk dict.
    Chunks that fail embedding are dropped.
    """
    enriched = []
    total    = len(chunks)

    for start in range(0, total, BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        texts = [c["text"] for c in batch]

        log.info(
            "Embedding batch %d-%d / %d ...",
            start + 1, min(start + BATCH_SIZE, total), total
        )

        embeddings = embed_texts(texts)
        if embeddings is None:
            log.error("Skipping batch %d-%d (all retries failed)", start, start + BATCH_SIZE)
            continue

        for chunk, emb in zip(batch, embeddings):
            enriched.append({**chunk, "embedding": emb})

    log.info("Embedding complete. %d / %d chunks embedded.", len(enriched), total)
    return enriched
