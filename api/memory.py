"""
memory.py
Short-term conversation memory with optional personalization.
Stores per-session history and extracted user preferences.
"""

import logging
import re
from collections import defaultdict
from datetime import datetime

log = logging.getLogger(__name__)

# In-memory store keyed by session_id
_sessions: dict[str, dict] = defaultdict(lambda: {
    "history":       [],       # list of {role, content} dicts
    "personalized":  False,    # did user opt in to personalization?
    "profile":       {},       # extracted facts about this user
    "created_at":    datetime.utcnow().isoformat(),
    "last_active":   datetime.utcnow().isoformat(),
})

MAX_HISTORY = 20   # messages to keep (10 turns)

# Patterns to extract user facts for personalization
PROFILE_PATTERNS = [
    (re.compile(r"\bi(?:'m| am) (?:a |an )?(\w+ student)", re.I), "status"),
    (re.compile(r"my (?:major|program) is ([^\.,\n]+)", re.I),      "program"),
    (re.compile(r"i(?:'m| am) (?:interested in|studying) ([^\.,\n]+)", re.I), "interest"),
    (re.compile(r"my advisor is ([A-Z][a-z]+ [A-Z][a-z]+)", re.I), "advisor"),
    (re.compile(r"i (?:start|started|joining) (?:in )?(\w+ \d{4})", re.I), "start_term"),
]


def get_session(session_id: str) -> dict:
    return _sessions[session_id]


def add_message(session_id: str, role: str, content: str) -> None:
    session = _sessions[session_id]
    session["history"].append({"role": role, "content": content})
    session["last_active"] = datetime.utcnow().isoformat()

    # Keep history bounded
    if len(session["history"]) > MAX_HISTORY:
        session["history"] = session["history"][-MAX_HISTORY:]

    # Extract facts if personalized
    if role == "user" and session["personalized"]:
        _extract_profile_facts(session, content)


def get_history(session_id: str) -> list[dict]:
    return _sessions[session_id]["history"]


def enable_personalization(session_id: str) -> str:
    _sessions[session_id]["personalized"] = True
    return (
        "Great! I'll remember things you share about yourself to give you "
        "more relevant answers. What would you like to know?"
    )


def is_personalized(session_id: str) -> bool:
    return _sessions[session_id]["personalized"]


def get_profile(session_id: str) -> dict:
    return _sessions[session_id]["profile"]


def _extract_profile_facts(session: dict, text: str) -> None:
    for pattern, key in PROFILE_PATTERNS:
        m = pattern.search(text)
        if m:
            session["profile"][key] = m.group(1).strip()
            log.info("Profile update [%s]: %s", key, session["profile"][key])


def build_personalized_context(session_id: str) -> str:
    """Return a short system note about the user for the LLM prompt."""
    profile = get_profile(session_id)
    if not profile:
        return ""
    parts = []
    if "status"   in profile: parts.append(f"User is a {profile['status']}")
    if "program"  in profile: parts.append(f"program: {profile['program']}")
    if "interest" in profile: parts.append(f"interested in: {profile['interest']}")
    if "advisor"  in profile: parts.append(f"advisor: {profile['advisor']}")
    if "start_term" in profile: parts.append(f"starting: {profile['start_term']}")
    return "User context: " + "; ".join(parts) + "." if parts else ""


def detect_personalization_request(query: str) -> bool:
    """Check if the user is asking to personalize their experience."""
    triggers = ["personalize", "remember me", "remember my", "save my", "track my"]
    lower = query.lower()
    return any(t in lower for t in triggers)


def clear_session(session_id: str) -> None:
    if session_id in _sessions:
        del _sessions[session_id]
