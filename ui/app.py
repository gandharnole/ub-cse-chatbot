"""
app.py
Streamlit frontend for the UB CSE BullBot chatbot.
Run with: streamlit run ui/app.py
"""

import json
import uuid
import requests
import streamlit as st

API_URL = "http://localhost:8000"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="BullBot — UB CSE Assistant",
    page_icon="🐂",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #005BBB 0%, #003d82 100%);
        padding: 1.2rem 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }
    .main-header h1 { margin: 0; font-size: 1.8rem; }
    .main-header p  { margin: 0.2rem 0 0; font-size: 0.95rem; opacity: 0.85; }

    .debug-section {
        background: #1e1e2e;
        color: #cdd6f4;
        border-radius: 8px;
        padding: 0.8rem 1rem;
        font-family: monospace;
        font-size: 0.8rem;
        margin-bottom: 0.5rem;
        overflow-x: auto;
    }
    .score-bar {
        background: #313244;
        border-radius: 4px;
        padding: 3px 8px;
        margin: 2px 0;
        display: flex;
        justify-content: space-between;
    }
    .score-high  { color: #a6e3a1; }
    .score-med   { color: #f9e2af; }
    .score-low   { color: #f38ba8; }

    .source-chip {
        display: inline-block;
        background: #e8f4f8;
        border: 1px solid #b0d4e8;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.75rem;
        margin: 2px;
        color: #005BBB;
        text-decoration: none;
    }
    .guardrail-pass { color: #a6e3a1; }
    .guardrail-fail { color: #f38ba8; }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────

if "session_id"   not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages"     not in st.session_state:
    st.session_state.messages = []
if "debug_log"    not in st.session_state:
    st.session_state.debug_log = []
if "show_debug"   not in st.session_state:
    st.session_state.show_debug = True

# ── Layout: two columns ───────────────────────────────────────────────────────

chat_col, debug_col = st.columns([2, 1], gap="medium")

# ── Left: Chat ────────────────────────────────────────────────────────────────

with chat_col:
    st.markdown("""
    <div class="main-header">
        <h1>🐂 BullBot</h1>
        <p>UB Computer Science & Engineering Assistant</p>
    </div>
    """, unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="🐂" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Ask me about UB CSE courses, faculty, programs..."):
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Call API
        with st.chat_message("assistant", avatar="🐂"):
            with st.spinner("Thinking..."):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={
                            "query":      prompt,
                            "session_id": st.session_state.session_id,
                        },
                        timeout=60,
                    )
                    resp.raise_for_status()
                    data = resp.json()

                    answer     = data["answer"]
                    debug_info = data.get("debug", {})

                    st.session_state.session_id = data["session_id"]
                    st.markdown(answer)

                    # Store for debug panel
                    st.session_state.debug_log.append({
                        "query": prompt,
                        "debug": debug_info,
                    })

                except requests.exceptions.ConnectionError:
                    answer = "⚠️ Cannot connect to the API. Make sure `uvicorn api.main:app --reload` is running."
                    st.error(answer)
                except Exception as e:
                    answer = f"⚠️ Error: {e}"
                    st.error(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})
        st.rerun()

# ── Right: Debug panel ────────────────────────────────────────────────────────

with debug_col:
    st.markdown("### 🔍 Debug panel")

    # Controls
    col1, col2 = st.columns(2)
    with col1:
        st.session_state.show_debug = st.toggle("Show debug", value=st.session_state.show_debug)
    with col2:
        if st.button("🗑 Clear chat"):
            try:
                requests.delete(f"{API_URL}/session/{st.session_state.session_id}", timeout=5)
            except Exception:
                pass
            st.session_state.messages  = []
            st.session_state.debug_log = []
            st.session_state.session_id = str(uuid.uuid4())
            st.rerun()

    # API stats
    try:
        stats = requests.get(f"{API_URL}/stats", timeout=3).json()
        chroma = stats.get("chroma", {})
        graph  = stats.get("graph",  {})
        st.caption(
            f"📦 {chroma.get('chunk_count', '?')} chunks  •  "
            f"🕸 {graph.get('nodes', '?')} graph nodes"
        )
    except Exception:
        st.caption("⚠️ API offline")

    st.divider()

    if not st.session_state.show_debug:
        st.info("Toggle debug on to see retrieval details.")
    elif not st.session_state.debug_log:
        st.info("Ask a question to see retrieval debug info here.")
    else:
        # Show most recent debug entry
        latest = st.session_state.debug_log[-1]
        query  = latest["query"]
        debug  = latest["debug"]

        st.markdown(f"**Query:** `{query[:60]}{'...' if len(query) > 60 else ''}`")

        # Guardrail
        passed = debug.get("guardrail_passed", True)
        status = "✅ passed" if passed else "❌ blocked"
        color  = "guardrail-pass" if passed else "guardrail-fail"
        st.markdown(f"**Guardrail:** <span class='{color}'>{status}</span>",
                    unsafe_allow_html=True)

        # KG lookup
        kg = debug.get("kg_lookup")
        if kg:
            found = "✅ found" if kg.get("found") else "➖ not found"
            st.markdown(f"**KG lookup:** `{kg.get('code')}` — {found}")

        # Retrieval scores
        retrieval = debug.get("retrieval", {})

        with st.expander("BM25 hits", expanded=False):
            bm25_hits = retrieval.get("bm25_hits", [])
            if bm25_hits:
                st.markdown('<div class="debug-section">', unsafe_allow_html=True)
                for h in bm25_hits[:5]:
                    score = h.get("score", 0)
                    cls   = "score-high" if score > 5 else "score-med" if score > 1 else "score-low"
                    text  = h["text"][:55].replace("\n", " ")
                    st.markdown(
                        f'<div class="score-bar">'
                        f'<span>{text}…</span>'
                        f'<span class="{cls}">{score:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption("No BM25 hits")

        with st.expander("Vector hits", expanded=False):
            vec_hits = retrieval.get("vector_hits", [])
            if vec_hits:
                st.markdown('<div class="debug-section">', unsafe_allow_html=True)
                for h in vec_hits[:5]:
                    score = h.get("score", 0)
                    cls   = "score-high" if score > 0.7 else "score-med" if score > 0.4 else "score-low"
                    text  = h["text"][:55].replace("\n", " ")
                    st.markdown(
                        f'<div class="score-bar">'
                        f'<span>{text}…</span>'
                        f'<span class="{cls}">{score:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption("No vector hits")

        with st.expander("Re-ranked chunks (sent to LLM)", expanded=True):
            reranked = retrieval.get("reranked", [])
            if reranked:
                st.markdown('<div class="debug-section">', unsafe_allow_html=True)
                for i, h in enumerate(reranked, 1):
                    score = h.get("ce_score", 0)
                    cls   = "score-high" if score > 3 else "score-med" if score > 0 else "score-low"
                    text  = h["text"][:55].replace("\n", " ")
                    st.markdown(
                        f'<div class="score-bar">'
                        f'<span>#{i} {text}…</span>'
                        f'<span class="{cls}">{score:.3f}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                st.markdown('</div>', unsafe_allow_html=True)
            else:
                st.caption("No re-ranked results")

        # Sources
        sources = debug.get("sources", [])
        if sources:
            st.markdown("**Sources:**")
            for url in sources:
                if url:
                    short = url.replace("https://engineering.buffalo.edu", "").replace("https://www.buffalo.edu", "")
                    st.markdown(
                        f'<a class="source-chip" href="{url}" target="_blank">{short[:50]}</a>',
                        unsafe_allow_html=True
                    )

        # Full debug JSON
        with st.expander("Raw debug JSON", expanded=False):
            st.json(debug)

        # History of all debug entries this session
        if len(st.session_state.debug_log) > 1:
            st.divider()
            st.markdown(f"**Session history:** {len(st.session_state.debug_log)} queries")
            for i, entry in enumerate(reversed(st.session_state.debug_log[:-1]), 1):
                with st.expander(f"Query {len(st.session_state.debug_log) - i}: {entry['query'][:40]}…"):
                    st.json(entry["debug"])
