"""
pdf_extractor.py - fixed UTF-8 writing
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from urllib.parse import urlparse

import requests
import fitz  # PyMuPDF

log = logging.getLogger(__name__)

OUTPUT_DIR  = Path("data/raw")
TIMEOUT_SEC = 30
DELAY_SEC   = 1.0

HEADERS = {
    "User-Agent": "UBCSEChatbotResearcher/1.0 (educational project)"
}


def url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def classify_pdf(url: str, text: str) -> str:
    lower = (url + " " + text[:500]).lower()
    if "syllabus" in lower or "syllab" in lower:
        return "syllabus"
    if "catalog" in lower or "bulletin" in lower:
        return "course_catalog"
    if "handbook" in lower:
        return "handbook"
    if "cse" in lower and any(c.isdigit() for c in lower):
        return "course_document"
    return "pdf_document"


def extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text_parts.append(page.get_text("text"))
    return "\n".join(text_parts).strip()


def download_and_extract(url: str, session: requests.Session, output_dir: Path) -> bool:
    out_path = output_dir / f"pdf_{url_id(url)}.json"
    if out_path.exists():
        log.debug("Already extracted: %s", url)
        return True

    try:
        resp = session.get(url, timeout=TIMEOUT_SEC, headers=HEADERS)
        resp.raise_for_status()

        ct = resp.headers.get("Content-Type", "")
        if "pdf" not in ct.lower() and not url.lower().endswith(".pdf"):
            log.warning("Not a PDF (Content-Type: %s): %s", ct, url)
            return False

        text = extract_pdf_text(resp.content)
        if not text.strip():
            log.warning("Empty PDF: %s", url)
            return False

        filename = Path(urlparse(url).path).name
        page_type = classify_pdf(url, text)

        doc = {
            "url":       url,
            "title":     filename,
            "page_type": page_type,
            "text":      text,
            "pdf_links": [],
        }

        # Always write as UTF-8 — fixes charmap errors on Windows
        out_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        log.info("[%s] PDF saved: %s → %s", page_type, filename, out_path.name)
        return True

    except Exception as e:
        log.warning("Failed PDF %s — %s", url, e)
        return False


def run_pdf_extraction(
    pdf_list_path: Path = Path("data/raw/_pdf_urls.json"),
    output_dir: Path = OUTPUT_DIR,
) -> None:
    if not pdf_list_path.exists():
        log.error("PDF URL list not found: %s. Run crawler first.", pdf_list_path)
        return

    urls = json.loads(pdf_list_path.read_text(encoding="utf-8"))
    log.info("Extracting %d PDFs...", len(urls))

    session = requests.Session()
    ok = 0
    for i, url in enumerate(urls, 1):
        log.info("PDF [%d/%d]: %s", i, len(urls), url)
        if download_and_extract(url, session, output_dir):
            ok += 1
        time.sleep(DELAY_SEC)

    log.info("PDF extraction complete. Success: %d/%d", ok, len(urls))
