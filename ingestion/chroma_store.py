"""
chroma_store.py
Loads embedded chunks into ChromaDB with metadata filtering support.
Also provides the retrieval interface used by the API.
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.config import Settings

log = logging.getLogger(__name__)

CHROMA_PATH       = Path("data/chroma_db")
COLLECTION_NAME   = "ub_cse"


def _chunk_id(chunk: dict) -> str:
    """Stable unique ID for a chunk."""
    key = f"{chunk['url']}::{chunk['chunk_index']}"
    return hashlib.md5(key.encode()).hexdigest()


def get_collection(path: Path = CHROMA_PATH) -> chromadb.Collection:
    """Return (or create) the ChromaDB collection."""
    path.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(path),
        settings=Settings(anonymized_telemetry=False),
    )
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def upsert_chunks(chunks: list[dict], path: Path = CHROMA_PATH) -> None:
    """
    Upsert embedded chunks into ChromaDB.
    chunks must have an 'embedding' field.
    """
    collection = get_collection(path)

    ids         = []
    embeddings  = []
    documents   = []
    metadatas   = []

    for chunk in chunks:
        if "embedding" not in chunk:
            continue

        cid = _chunk_id(chunk)
        ids.append(cid)
        embeddings.append(chunk["embedding"])
        documents.append(chunk["text"])
        metadatas.append({
            "url":          chunk.get("url", ""),
            "title":        chunk.get("title", ""),
            "page_type":    chunk.get("page_type", "general"),
            "course_codes": ",".join(chunk.get("course_codes", [])),
            "chunk_index":  chunk.get("chunk_index", 0),
        })

    # Upsert in batches of 500 (ChromaDB limit)
    batch_size = 500
    for i in range(0, len(ids), batch_size):
        collection.upsert(
            ids=ids[i : i + batch_size],
            embeddings=embeddings[i : i + batch_size],
            documents=documents[i : i + batch_size],
            metadatas=metadatas[i : i + batch_size],
        )
        log.info("Upserted %d / %d chunks", min(i + batch_size, len(ids)), len(ids))

    log.info("ChromaDB upsert complete. Total: %d chunks in collection '%s'",
             len(ids), COLLECTION_NAME)


def semantic_search(
    query_embedding: list[float],
    n_results:       int = 10,
    page_type:       Optional[str] = None,
    path:            Path = CHROMA_PATH,
) -> list[dict]:
    """
    Vector similarity search. Returns list of result dicts
    with 'text', 'metadata', and 'distance'.
    """
    collection = get_collection(path)

    where = {"page_type": page_type} if page_type else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        hits.append({
            "text":     doc,
            "metadata": meta,
            "score":    1 - dist,   # cosine similarity (higher = better)
        })

    return hits


def get_stats(path: Path = CHROMA_PATH) -> dict:
    """Return collection stats."""
    collection = get_collection(path)
    count = collection.count()
    return {"collection": COLLECTION_NAME, "chunk_count": count}
