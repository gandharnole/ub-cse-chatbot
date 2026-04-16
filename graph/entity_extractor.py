"""
entity_extractor.py
Extracts structured entities (courses, faculty, labs) from cleaned JSON docs.
"""

import json
import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

# ── Regex patterns ────────────────────────────────────────────────────────────

COURSE_CODE_RE  = re.compile(r"\bCSE\s*(\d{3,4})\b", re.IGNORECASE)
COURSE_TITLE_RE = re.compile(
    r"CSE\s*\d{3,4}[^\n]*?[–\-:]\s*([A-Z][^\n]{5,60})", re.IGNORECASE
)
PREREQ_RE = re.compile(
    r"[Pp]re-?requisite[s]?\s*[:\-]?\s*([^\n.]{5,120})"
)
CREDIT_RE = re.compile(r"(\d)\s*credit(?:\s*hour)?s?", re.IGNORECASE)

# Faculty name heuristic: "First Last" or "Last, First" in faculty pages
FACULTY_NAME_RE = re.compile(
    r"\b([A-Z][a-z]+(?:\s[A-Z][a-z]+){1,2})\b"
)

LAB_KEYWORDS = [
    "laboratory", "lab", "center for", "institute for",
    "research group", "research center", "group for"
]

RESEARCH_AREAS = [
    "artificial intelligence", "machine learning", "computer vision",
    "natural language processing", "cybersecurity", "cryptography",
    "databases", "data science", "systems", "networking",
    "high performance computing", "bioinformatics", "robotics",
    "human computer interaction", "programming languages",
    "software engineering", "algorithms", "theory",
    "mobile computing", "edge computing", "information integrity",
]


def extract_courses(doc: dict) -> list[dict]:
    text      = doc.get("text", "")
    url       = doc.get("url", "")
    page_type = doc.get("page_type", "")
    courses   = []

    for m in COURSE_CODE_RE.finditer(text):
        number = m.group(1)
        code   = f"CSE {number}"

        # Try to grab a title on the same line
        line_start = text.rfind("\n", 0, m.start()) + 1
        line_end   = text.find("\n", m.end())
        line       = text[line_start:line_end if line_end > 0 else line_start + 200]

        title = ""
        title_m = re.search(r"[–\-:]\s*([A-Z][^\n]{5,60})", line)
        if title_m:
            title = title_m.group(1).strip()

        # Prerequisites in nearby text
        context = text[m.start():m.start() + 400]
        prereqs = []
        for pm in PREREQ_RE.finditer(context):
            raw = pm.group(1).strip()
            codes = COURSE_CODE_RE.findall(raw)
            prereqs = [f"CSE {c}" for c in codes]

        # Credits
        credits = None
        credit_m = CREDIT_RE.search(context)
        if credit_m:
            credits = int(credit_m.group(1))

        courses.append({
            "code":    code,
            "number":  number,
            "title":   title,
            "prereqs": prereqs,
            "credits": credits,
            "source":  url,
        })

    return courses


def extract_faculty(doc: dict) -> list[dict]:
    text      = doc.get("text", "")
    url       = doc.get("url", "")
    page_type = doc.get("page_type", "")

    if page_type not in ("faculty", "general") and "faculty" not in url.lower():
        return []

    # Extract name from URL slug (most reliable for profile pages)
    faculty = []
    slug_m = re.search(r"/profiles/faculty/(?:ladder|teaching|affiliated|emeriti)/([^/.]+)", url)
    if slug_m:
        slug  = slug_m.group(1)
        parts = [p.capitalize() for p in slug.replace("-", " ").split()]
        name  = " ".join(parts)

        # Research areas from text
        text_lower = text.lower()
        areas = [a for a in RESEARCH_AREAS if a in text_lower]

        # Courses taught — look for CSE codes
        taught = list({f"CSE {m}" for m in COURSE_CODE_RE.findall(text)})

        # Email heuristic
        email_m = re.search(r"[\w.\-]+@(?:buffalo\.edu|cse\.buffalo\.edu)", text)
        email   = email_m.group(0) if email_m else ""

        faculty.append({
            "name":            name,
            "url":             url,
            "research_areas":  areas,
            "courses_taught":  taught,
            "email":           email,
        })

    return faculty


def extract_labs(doc: dict) -> list[dict]:
    text = doc.get("text", "")
    url  = doc.get("url", "")
    labs = []

    # Only extract from research/general pages, not PDFs or faculty profiles
    page_type = doc.get("page_type", "")
    if page_type not in ("research", "general"):
        return []

    lines = text.splitlines()
    for line in lines:
        stripped = line.strip()
        lower    = stripped.lower()

        # Must contain a strong lab/center keyword AND look like a proper name
        strong_kws = ["laboratory", "center for", "institute for", "research center", "research group"]
        if not any(kw in lower for kw in strong_kws):
            continue

        # Reject short/long lines and lines that are clearly sentences
        if len(stripped) < 10 or len(stripped) > 100:
            continue
        if stripped.endswith((".", ",", ";")):
            continue
        # Must start with a capital letter (proper name)
        if not stripped[0].isupper():
            continue

        areas = [a for a in RESEARCH_AREAS if a in lower]
        labs.append({
            "name":           stripped,
            "research_areas": areas,
            "source":         url,
        })

    return labs


def extract_all(raw_dir: Path = RAW_DIR) -> dict:
    """
    Returns:
        {
          "courses": [...],
          "faculty": [...],
          "labs":    [...],
        }
    """
    all_courses = {}
    all_faculty = {}
    all_labs    = []

    files = [f for f in raw_dir.glob("*.json") if not f.name.startswith("_")]
    log.info("Extracting entities from %d files...", len(files))

    for path in files:
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        for course in extract_courses(doc):
            code = course["code"]
            if code not in all_courses:
                all_courses[code] = course
            else:
                # Merge — prefer richer entries
                existing = all_courses[code]
                if not existing["title"] and course["title"]:
                    existing["title"] = course["title"]
                if not existing["prereqs"] and course["prereqs"]:
                    existing["prereqs"] = course["prereqs"]
                if not existing["credits"] and course["credits"]:
                    existing["credits"] = course["credits"]

        for fac in extract_faculty(doc):
            name = fac["name"]
            if name not in all_faculty:
                all_faculty[name] = fac
            else:
                existing = all_faculty[name]
                existing["research_areas"] = list(
                    set(existing["research_areas"] + fac["research_areas"])
                )
                existing["courses_taught"] = list(
                    set(existing["courses_taught"] + fac["courses_taught"])
                )

        all_labs.extend(extract_labs(doc))

    # Deduplicate labs by name
    seen_labs = set()
    unique_labs = []
    for lab in all_labs:
        if lab["name"] not in seen_labs:
            seen_labs.add(lab["name"])
            unique_labs.append(lab)

    result = {
        "courses": list(all_courses.values()),
        "faculty": list(all_faculty.values()),
        "labs":    unique_labs,
    }

    log.info(
        "Extraction done. Courses: %d | Faculty: %d | Labs: %d",
        len(result["courses"]), len(result["faculty"]), len(result["labs"])
    )
    return result
