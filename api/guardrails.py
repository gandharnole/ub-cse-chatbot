"""
guardrails.py
Classifies queries as in-scope (UB CSE) or out-of-scope.
Uses a keyword + embedding heuristic — fast, no extra model needed.
"""

import re
import logging

log = logging.getLogger(__name__)

# Strong signals that a query IS about UB CSE
IN_SCOPE_KEYWORDS = [
    "cse", "computer science", "computer engineering",
    "ub ", "buffalo", "ubit", "seas",
    "professor", "faculty", "advisor", "advisement",
    "course", "class", "credits", "prerequisite", "syllabus",
    "graduate", "undergraduate", "ms", "phd", "bs", "bachelor", "master", "doctor",
    "admission", "apply", "application", "gre", "gpa", "toefl", "ielts",
    "research", "lab", "thesis", "dissertation",
    "internship", "career", "scholarship", "assistantship", "fellowship",
    "tuition", "financial aid",
    "registration", "enrollment", "hub", "degree",
]

# Strong signals that a query is OUT of scope
OUT_OF_SCOPE_KEYWORDS = [
    "pizza", "restaurant", "food", "recipe",
    "weather", "forecast",
    "stock", "crypto", "bitcoin",
    "movie", "film", "netflix", "spotify",
    "sports", "nfl", "nba", "soccer",
    "politics", "election", "president",
    "joke", "funny",
    "dating", "relationship",
    "medical advice", "symptoms", "diagnosis",
]

# These patterns are almost always CSE-related
COURSE_CODE_RE = re.compile(r"\bcse\s*\d{3,4}\b", re.IGNORECASE)
FACULTY_TITLE_RE = re.compile(
    r"\b(professor|prof\.|dr\.|ph\.d)\b", re.IGNORECASE
)

REDIRECT_MESSAGE = (
    "I'm BullBot, the UB CSE assistant, and I can only help with questions about "
    "the UB Computer Science and Engineering department — programs, courses, faculty, "
    "admissions, and research. Could you ask me something related to UB CSE?"
)


def is_in_scope(query: str) -> tuple[bool, str]:
    """
    Returns (in_scope: bool, reason: str).
    Reason is used for the debug panel in the UI.
    """
    lower = query.lower()

    # Hard in-scope: contains a CSE course code or faculty title
    if COURSE_CODE_RE.search(lower):
        return True, "matched CSE course code"
    if FACULTY_TITLE_RE.search(lower):
        return True, "matched faculty title keyword"

    # Count keyword hits
    in_hits  = sum(1 for kw in IN_SCOPE_KEYWORDS  if kw in lower)
    out_hits = sum(1 for kw in OUT_OF_SCOPE_KEYWORDS if kw in lower)

    if out_hits > 0 and in_hits == 0:
        return False, f"matched out-of-scope keywords: {out_hits}"

    if in_hits >= 1:
        return True, f"matched {in_hits} in-scope keyword(s)"

    # Short greetings and small talk — allow with a gentle redirect built in
    if len(query.split()) <= 5:
        return True, "short query — allowed with context check"

    # Default: allow and let the LLM decide based on context
    return True, "no strong signal — defaulting to in-scope"


def check_and_respond(query: str) -> tuple[bool, str | None]:
    """
    Returns (should_proceed, redirect_message_or_None).
    If should_proceed is False, redirect_message contains the canned response.
    """
    in_scope, reason = is_in_scope(query)
    log.info("Guardrail check: in_scope=%s reason='%s'", in_scope, reason)

    if not in_scope:
        return False, REDIRECT_MESSAGE
    return True, None
