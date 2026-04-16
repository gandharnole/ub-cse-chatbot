"""
llm.py
Ollama client for chat generation using llama3.2.
"""

import logging
from typing import Iterator

import ollama

log = logging.getLogger(__name__)

LLM_MODEL = "llama3.2"

SYSTEM_PROMPT = """You are BullBot, the official AI assistant for the University at Buffalo \
Department of Computer Science and Engineering (UB CSE).

Your job is to answer questions about:
- UB CSE undergraduate and graduate programs (BS, MS, PhD)
- Course descriptions, prerequisites, and schedules
- Faculty members, their research, and office hours
- Admissions requirements and application process
- Research labs, centers, and groups
- Departmental policies and resources
- Student organizations and events

Guidelines:
- Answer ONLY from the provided context. Do not hallucinate facts.
- If the context does not contain enough information, say so clearly.
- Be concise, friendly, and helpful.
- When mentioning courses, always include the course code (e.g. CSE 574).
- When mentioning faculty, include their full name.
- Never answer questions unrelated to UB CSE — politely redirect instead.
"""


def generate(
    query:   str,
    context: str,
    history: list[dict] | None = None,
    stream:  bool = False,
) -> str:
    """
    Generate a response given a query and retrieved context.

    Args:
        query:   The user's question.
        context: Retrieved chunks concatenated as a string.
        history: List of {"role": "user"/"assistant", "content": "..."} dicts.
        stream:  If True, prints tokens as they arrive (for CLI testing).

    Returns:
        The assistant's response as a string.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-6:])  # last 3 turns (6 messages)

    user_content = f"""Context from UB CSE website:
---
{context}
---

Question: {query}

Answer based only on the context above."""

    messages.append({"role": "user", "content": user_content})

    try:
        if stream:
            full = ""
            for chunk in ollama.chat(model=LLM_MODEL, messages=messages, stream=True):
                token = chunk["message"]["content"]
                print(token, end="", flush=True)
                full += token
            print()
            return full
        else:
            response = ollama.chat(model=LLM_MODEL, messages=messages)
            return response["message"]["content"]
    except Exception as e:
        log.error("LLM generation failed: %s", e)
        return "I'm sorry, I encountered an error generating a response. Please try again."
