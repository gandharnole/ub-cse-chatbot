"""
retriever.py
Hybrid retrieval pipeline:
  1. BM25 keyword search  (exact match, good for course codes)
  2. Semantic vector search (ChromaDB + nomic-embed-text)
  3. Score fusion (RRF)
  4. Cross-encoder re-ranking (sentence-transformers)
"""

import json
import logging
import re
from pathlib import Path

import ollama
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from ingestion.chroma_store import semantic_search

log = logging.getLogger(__name__)

CHUNKS_PATH    = Path("data/chunks/all_chunks.json")
EMBED_MODEL    = "nomic-embed-text"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K_RETRIEVE = 10   # candidates before re-ranking
TOP_K_FINAL    = 4    # chunks passed to LLM

# ── BM25 index (built once at startup) ───────────────────────────────────────

_bm25_index:  BM25Okapi | None = None
_bm25_chunks: list[dict]       = []
_reranker:    CrossEncoder | None = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def load_indexes() -> None:
    """Load BM25 index and cross-encoder. Called once at API startup."""
    global _bm25_index, _bm25_chunks, _reranker

    log.info("Loading BM25 index from %s ...", CHUNKS_PATH)
    _bm25_chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
    corpus = [_tokenize(c["text"]) for c in _bm25_chunks]
    _bm25_index = BM25Okapi(corpus)
    log.info("BM25 index ready. Corpus size: %d chunks", len(_bm25_chunks))

    log.info("Loading cross-encoder reranker: %s ...", RERANKER_MODEL)
    _reranker = CrossEncoder(RERANKER_MODEL)
    log.info("Reranker ready.")


# ── Retrieval steps ───────────────────────────────────────────────────────────

def bm25_search(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    if _bm25_index is None:
        return []
    scores = _bm25_index.get_scores(_tokenize(query))
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    results = []
    for i in top_indices:
        if scores[i] > 0:
            results.append({
                "text":     _bm25_chunks[i]["text"],
                "metadata": {
                    "url":       _bm25_chunks[i].get("url", ""),
                    "title":     _bm25_chunks[i].get("title", ""),
                    "page_type": _bm25_chunks[i].get("page_type", ""),
                },
                "bm25_score": float(scores[i]),
                "score":      0.0,  # filled after fusion
            })
    return results


def vector_search(query: str, top_k: int = TOP_K_RETRIEVE) -> list[dict]:
    try:
        response = ollama.embed(model=EMBED_MODEL, input=[query])
        embedding = response.embeddings[0]
    except Exception as e:
        log.error("Embedding failed: %s", e)
        return []
    return semantic_search(embedding, n_results=top_k)


def reciprocal_rank_fusion(
    bm25_results:   list[dict],
    vector_results: list[dict],
    k: int = 60,
) -> list[dict]:
    """
    Fuse two ranked lists using Reciprocal Rank Fusion.
    Returns a merged, deduplicated list sorted by RRF score.
    """
    scores: dict[str, float] = {}
    docs:   dict[str, dict]  = {}

    for rank, result in enumerate(bm25_results):
        key = result["text"][:100]
        scores[key]  = scores.get(key, 0) + 1 / (k + rank + 1)
        docs[key]    = result

    for rank, result in enumerate(vector_results):
        key = result["text"][:100]
        scores[key]  = scores.get(key, 0) + 1 / (k + rank + 1)
        if key not in docs:
            docs[key] = result

    fused = sorted(docs.values(), key=lambda d: scores[d["text"][:100]], reverse=True)
    for doc in fused:
        doc["rrf_score"] = scores[doc["text"][:100]]
    return fused


def rerank(query: str, candidates: list[dict], top_k: int = TOP_K_FINAL) -> list[dict]:
    """
    Re-rank candidates using a cross-encoder.
    Returns top_k results with cross-encoder scores added.
    """
    if _reranker is None or not candidates:
        return candidates[:top_k]

    pairs  = [(query, c["text"]) for c in candidates]
    ce_scores = _reranker.predict(pairs)

    for candidate, score in zip(candidates, ce_scores):
        candidate["ce_score"] = float(score)

    reranked = sorted(candidates, key=lambda c: c["ce_score"], reverse=True)

    log.info("Re-ranking scores: %s",
             [round(c["ce_score"], 3) for c in reranked[:top_k]])

    return reranked[:top_k]


# ── Public interface ──────────────────────────────────────────────────────────

def retrieve(query: str) -> tuple[list[dict], dict]:
    """
    Full hybrid retrieval pipeline.

    Returns:
        (final_chunks, debug_info)
        debug_info contains scores at each stage for the UI panel.
    """
    debug = {}

    # Step 1: BM25
    bm25_hits = bm25_search(query, top_k=TOP_K_RETRIEVE)
    debug["bm25_hits"] = [
        {"text": h["text"][:80], "score": round(h.get("bm25_score", 0), 4)}
        for h in bm25_hits
    ]

    # Step 2: Vector
    vec_hits = vector_search(query, top_k=TOP_K_RETRIEVE)
    debug["vector_hits"] = [
        {"text": h["text"][:80], "score": round(h.get("score", 0), 4)}
        for h in vec_hits
    ]

    # Step 3: RRF fusion
    fused = reciprocal_rank_fusion(bm25_hits, vec_hits)
    debug["fused_count"] = len(fused)

    # Step 4: Cross-encoder re-rank
    final = rerank(query, fused, top_k=TOP_K_FINAL)
    debug["reranked"] = [
        {"text": c["text"][:80], "ce_score": round(c.get("ce_score", 0), 4)}
        for c in final
    ]

    return final, debug


def format_context(chunks: list[dict]) -> str:
    """Concatenate chunk texts into a single context string for the LLM."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        url   = chunk.get("metadata", {}).get("url", "")
        title = chunk.get("metadata", {}).get("title", "")
        src   = title or url
        parts.append(f"[{i}] {chunk['text']}\nSource: {src}")
    return "\n\n".join(parts)
