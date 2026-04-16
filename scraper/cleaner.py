"""
cleaner.py - fixed encoding handling
"""

import json
import re
import logging
from pathlib import Path

log = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")

BOILERPLATE = [
    "skip to main content",
    "university at buffalo",
    "the state university of new york",
    "accessibility",
    "privacy policy",
    "contact us",
    "search this site",
    "follow us on",
    "facebook",
    "twitter",
    "instagram",
    "linkedin",
    "youtube",
    "copyright",
    "all rights reserved",
    "back to top",
]

COURSE_CODE_RE = re.compile(r"\bCSE\s*\d{3,4}\b", re.IGNORECASE)


def clean_text(text: str) -> str:
    lines = text.splitlines()
    cleaned = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if any(bp in lower for bp in BOILERPLATE):
            continue
        stripped = re.sub(r"[ \t]+", " ", stripped)
        cleaned.append(stripped)
    deduped = []
    prev = None
    for line in cleaned:
        if line != prev:
            deduped.append(line)
        prev = line
    return "\n".join(deduped)


def extract_course_codes(text: str) -> list:
    matches = COURSE_CODE_RE.findall(text)
    normalised = []
    for m in matches:
        m = re.sub(r"(?<=CSE)(\s*)(?=\d)", " ", m, flags=re.IGNORECASE).upper()
        normalised.append(m.strip())
    return list(dict.fromkeys(normalised))


def enrich_metadata(doc: dict) -> dict:
    text = doc.get("text", "")
    doc["course_codes"] = extract_course_codes(text)
    doc["char_count"]   = len(text)
    doc["word_count"]   = len(text.split())
    return doc


def process_file(path: Path) -> bool:
    # Try multiple encodings
    raw = None
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            raw = json.loads(path.read_text(encoding=enc))
            break
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue

    if raw is None:
        log.warning("Could not read %s with any encoding", path.name)
        return False

    raw["text"] = clean_text(raw.get("text", ""))
    if not raw["text"].strip():
        log.debug("Empty after cleaning, skipping: %s", path.name)
        return False

    raw = enrich_metadata(raw)
    path.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    return True


def run_cleaning(raw_dir: Path = RAW_DIR) -> None:
    files = [f for f in raw_dir.glob("*.json") if not f.name.startswith("_")]
    log.info("Cleaning %d raw files...", len(files))
    ok = sum(1 for f in files if process_file(f))
    log.info("Cleaning complete. Processed: %d/%d", ok, len(files))
