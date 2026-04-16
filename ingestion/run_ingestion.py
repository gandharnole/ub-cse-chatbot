"""
run_ingestion.py
Runs the full ingestion pipeline: chunk → embed → store in ChromaDB.

Usage:
    python -m ingestion.run_ingestion            # full pipeline
    python -m ingestion.run_ingestion --chunk    # chunk only
    python -m ingestion.run_ingestion --embed    # chunk + embed + store
    python -m ingestion.run_ingestion --stats    # print DB stats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ingestion.chunker      import run_chunking
from ingestion.embedder     import embed_chunks
from ingestion.chroma_store import upsert_chunks, get_stats

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/ingestion.log", mode="w"),
    ]
)
log = logging.getLogger(__name__)

CHUNKS_PATH = Path("data/chunks/all_chunks.json")


def main():
    parser = argparse.ArgumentParser(description="UB CSE ingestion pipeline")
    parser.add_argument("--chunk",  action="store_true")
    parser.add_argument("--embed",  action="store_true")
    parser.add_argument("--stats",  action="store_true")
    args = parser.parse_args()

    run_all = not any([args.chunk, args.embed, args.stats])

    if args.stats:
        stats = get_stats()
        log.info("ChromaDB stats: %s", stats)
        return

    # ── Step 1: Chunk ────────────────────────────────────────────────────────
    if run_all or args.chunk or args.embed:
        log.info("=" * 50)
        log.info("STEP 1 — Chunking")
        log.info("=" * 50)
        chunks = run_chunking()
        log.info("Total chunks: %d", len(chunks))

    # ── Step 2: Embed ────────────────────────────────────────────────────────
    if run_all or args.embed:
        log.info("=" * 50)
        log.info("STEP 2 — Embedding")
        log.info("=" * 50)

        # Load from disk if we skipped chunking step
        if not (run_all or args.chunk):
            if not CHUNKS_PATH.exists():
                log.error("No chunks found. Run --chunk first.")
                return
            chunks = json.loads(CHUNKS_PATH.read_text(encoding="utf-8"))
            log.info("Loaded %d chunks from disk", len(chunks))

        embedded = embed_chunks(chunks)

        # ── Step 3: Store ────────────────────────────────────────────────────
        log.info("=" * 50)
        log.info("STEP 3 — Storing in ChromaDB")
        log.info("=" * 50)
        upsert_chunks(embedded)

        stats = get_stats()
        log.info("Final DB stats: %s", stats)

    log.info("Ingestion complete.")


if __name__ == "__main__":
    main()
