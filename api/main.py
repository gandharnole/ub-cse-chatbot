"""
main.py
FastAPI backend for the UB CSE chatbot.

Endpoints:
  POST /chat          — main chat endpoint
  GET  /health        — health check
  GET  /stats         — DB + graph stats
  DELETE /session     — clear a session
"""

import logging
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.retriever  import retrieve, format_context, load_indexes
from api.llm        import generate
from api.guardrails import check_and_respond
from api.memory     import (
    add_message, get_history, get_session,
    enable_personalization, is_personalized,
    build_personalized_context, detect_personalization_request,
    clear_session,
)
from ingestion.chroma_store import get_stats
from graph.kg_builder       import load_graph, get_course_info, suggest_related

import re
from pathlib import Path

log = logging.getLogger(__name__)

# ── Startup ───────────────────────────────────────────────────────────────────

_graph = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _graph
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    log.info("Loading retrieval indexes...")
    load_indexes()
    log.info("Loading knowledge graph...")
    kg_path = Path("graph/kg_store.json")
    if kg_path.exists():
        _graph = load_graph(kg_path)
        log.info("Knowledge graph loaded.")
    log.info("BullBot API ready.")
    yield


app = FastAPI(title="UB CSE Chatbot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query:      str
    session_id: str | None = None

class ChatResponse(BaseModel):
    answer:     str
    session_id: str
    debug:      dict


COURSE_CODE_RE = re.compile(r"\bCSE\s*(\d{3,4})\b", re.IGNORECASE)

# ── Chat endpoint ─────────────────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    query      = req.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    debug: dict = {"session_id": session_id}

    # ── 1. Personalization opt-in check ──────────────────────────────────────
    if detect_personalization_request(query):
        msg = enable_personalization(session_id)
        add_message(session_id, "user",      query)
        add_message(session_id, "assistant", msg)
        return ChatResponse(answer=msg, session_id=session_id,
                            debug={"personalization": "enabled"})

    # ── 2. Guardrail check ────────────────────────────────────────────────────
    proceed, redirect = check_and_respond(query)
    debug["guardrail_passed"] = proceed
    if not proceed:
        add_message(session_id, "user",      query)
        add_message(session_id, "assistant", redirect)
        return ChatResponse(answer=redirect, session_id=session_id, debug=debug)

    # ── 3. Knowledge graph lookup (bonus: course → faculty + labs) ────────────
    kg_context = ""
    course_m   = COURSE_CODE_RE.search(query)
    if course_m and _graph:
        code     = f"CSE {course_m.group(1)}"
        info     = get_course_info(_graph, code)
        related  = suggest_related(_graph, code)
        if info:
            kg_context = (
                f"\nKnowledge graph — {code}:\n"
                f"  Title:      {info.get('title', 'N/A')}\n"
                f"  Credits:    {info.get('credits', 'N/A')}\n"
                f"  Prereqs:    {', '.join(info.get('prereqs', [])) or 'None'}\n"
                f"  Taught by:  {', '.join(related.get('faculty', [{}])[0:3] and [f['name'] for f in related.get('faculty', [])])}\n"
                f"  Related labs: {', '.join(related.get('labs', [])[:3]) or 'None'}\n"
            )
        debug["kg_lookup"] = {"code": code, "found": bool(info)}

    # ── 4. Hybrid retrieval ───────────────────────────────────────────────────
    chunks, retrieval_debug = retrieve(query)
    debug["retrieval"] = retrieval_debug

    # ── 5. Build context ──────────────────────────────────────────────────────
    context = format_context(chunks)
    if kg_context:
        context = kg_context + "\n\n" + context

    # Add personalization note
    personal_note = build_personalized_context(session_id) if is_personalized(session_id) else ""
    if personal_note:
        context = personal_note + "\n\n" + context

    # ── 6. Generate answer ────────────────────────────────────────────────────
    history = get_history(session_id)
    answer  = generate(query=query, context=context, history=history)

    # ── 7. Update memory ──────────────────────────────────────────────────────
    add_message(session_id, "user",      query)
    add_message(session_id, "assistant", answer)

    debug["sources"] = [
        c.get("metadata", {}).get("url", "") for c in chunks
    ]

    return ChatResponse(answer=answer, session_id=session_id, debug=debug)


# ── Utility endpoints ─────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model": "llama3.2"}


@app.get("/stats")
async def stats():
    db_stats = get_stats()
    graph_stats = {}
    if _graph:
        graph_stats = {
            "nodes": _graph.number_of_nodes(),
            "edges": _graph.number_of_edges(),
        }
    return {"chroma": db_stats, "graph": graph_stats}


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    clear_session(session_id)
    return {"cleared": session_id}
