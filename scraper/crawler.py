"""
crawler.py - with dynamic content fallback and UTF-8 safe writing
"""

import json
import time
import hashlib
import logging
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SEED_URLS = [
    "https://engineering.buffalo.edu/computer-science-engineering.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate.html",
    "https://engineering.buffalo.edu/computer-science-engineering/undergraduate.html",
    "https://engineering.buffalo.edu/computer-science-engineering/people.html",
    "https://engineering.buffalo.edu/computer-science-engineering/research.html",
    "https://engineering.buffalo.edu/computer-science-engineering/people/faculty-directory.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/courses/course-descriptions.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/ms-in-computer-science-and-engineering.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/degrees-and-programs/phd-in-computer-science-and-engineering.html",
    "https://engineering.buffalo.edu/computer-science-engineering/undergraduate/degrees-and-programs/bs-in-computer-science.html",
    "https://engineering.buffalo.edu/computer-science-engineering/undergraduate/degrees-and-programs/bs-in-computer-engineering.html",
    "https://engineering.buffalo.edu/computer-science-engineering/graduate/admissions.html",
    "https://engineering.buffalo.edu/computer-science-engineering/research/research-centers-institutes-labs-and-groups.html",
]

ALLOWED_DOMAINS = {
    "engineering.buffalo.edu",
    "www.buffalo.edu",
    "catalog.buffalo.edu",
}

ALLOWED_PATH_PREFIXES = [
    "/computer-science-engineering",
]

SKIP_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".gif", ".svg",
                   ".zip", ".doc", ".docx", ".ppt", ".pptx"}

MAX_PAGES   = 500
DELAY_SEC   = 0.5
TIMEOUT_SEC = 15
OUTPUT_DIR  = Path("data/raw")

HEADERS = {
    "User-Agent": "UBCSEChatbotResearcher/1.0 (educational project)"
}


def url_id(url: str) -> str:
    return hashlib.md5(url.encode()).hexdigest()[:12]


def is_allowed(url: str) -> bool:
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    if parsed.netloc not in ALLOWED_DOMAINS:
        return False
    ext = Path(parsed.path).suffix.lower()
    if ext in SKIP_EXTENSIONS:
        return False
    # Only crawl CSE pages — tighter scope = better quality data
    if parsed.netloc == "engineering.buffalo.edu":
        return any(parsed.path.startswith(p) for p in ALLOWED_PATH_PREFIXES)
    return False  # skip www.buffalo.edu noise


def classify_page(url: str, soup: BeautifulSoup) -> str:
    path  = urlparse(url).path.lower()
    title = (soup.title.string or "").lower() if soup.title else ""
    text  = title + " " + path
    if "faculty" in text or "people" in text or "directory" in text or "profile" in text:
        return "faculty"
    if "course" in text or "cse 4" in text or "cse 5" in text or "catalog" in text or "description" in text:
        return "course"
    if "graduate" in text or "ms " in text or "phd" in text:
        return "graduate_program"
    if "undergraduate" in text or "bs " in text:
        return "undergraduate_program"
    if "research" in text or "lab" in text or "center" in text:
        return "research"
    if "admission" in text or "apply" in text:
        return "admissions"
    return "general"


def extract_text(soup: BeautifulSoup) -> str:
    """Extract text — tries main content area first, falls back to full body."""
    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "noscript"]):
        tag.decompose()

    # Try progressively broader selectors
    selectors = [
        soup.find("main"),
        soup.find("div", {"id": "page-content"}),
        soup.find("div", {"id": "main-content"}),
        soup.find("article"),
        soup.find("div", {"class": lambda c: c and any(
            x in " ".join(c) for x in ["content", "main", "body", "text"]
        )}),
        soup.body,
    ]

    container = next((s for s in selectors if s is not None), None)
    if container is None:
        return ""

    lines = []
    for elem in container.find_all(["h1","h2","h3","h4","h5","p","li","td","th","span","div"]):
        # Only grab leaf-ish elements with meaningful text
        children_text = elem.get_text(separator=" ", strip=True)
        if children_text and len(children_text) > 20:
            lines.append(children_text)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)

    return "\n".join(unique)


def extract_links(base_url: str, soup: BeautifulSoup) -> list:
    links = []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        absolute = urljoin(base_url, href).split("#")[0]
        if absolute and is_allowed(absolute):
            links.append(absolute)
    return list(set(links))


def extract_pdf_links(base_url: str, soup: BeautifulSoup) -> list:
    pdfs = []
    for a in soup.find_all("a", href=True):
        href = urljoin(base_url, a["href"])
        if href.lower().endswith(".pdf"):
            pdfs.append(href)
    return list(set(pdfs))


class Crawler:
    def __init__(self, output_dir: Path = OUTPUT_DIR):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visited:  set = set()
        self.queue:    list = []
        self.pdf_urls: list = []
        self.saved = 0
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def _fetch(self, url: str) -> Optional[BeautifulSoup]:
        try:
            resp = self.session.get(url, timeout=TIMEOUT_SEC, allow_redirects=True)
            resp.raise_for_status()
            ct = resp.headers.get("Content-Type", "")
            if "text/html" not in ct:
                return None
            # Use detected encoding, fall back to latin-1 to avoid decode errors
            resp.encoding = resp.apparent_encoding or "latin-1"
            return BeautifulSoup(resp.text, "html.parser")
        except Exception as e:
            log.warning("Failed to fetch %s — %s", url, e)
            return None

    def _save(self, url: str, soup: BeautifulSoup) -> bool:
        page_type = classify_page(url, soup)
        text      = extract_text(soup)
        pdf_links = extract_pdf_links(url, soup)
        title     = (soup.title.string or "").strip() if soup.title else ""

        if not text.strip():
            log.debug("Empty text, skipping: %s", url)
            return False

        doc = {
            "url":       url,
            "title":     title,
            "page_type": page_type,
            "text":      text,
            "pdf_links": pdf_links,
        }

        out_path = self.output_dir / f"{url_id(url)}.json"
        out_path.write_text(
            json.dumps(doc, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )
        self.saved += 1
        log.info("[%s] Saved (#%d): %s", page_type, self.saved, url[:80])
        self.pdf_urls.extend(pdf_links)
        return True

    def run(self, seeds: list = SEED_URLS) -> None:
        self.queue = list(seeds)
        count = 0

        log.info("Starting crawl with %d seed URLs (scope: CSE pages only)", len(seeds))

        while self.queue and count < MAX_PAGES:
            url = self.queue.pop(0)
            if url in self.visited:
                continue
            self.visited.add(url)
            count += 1

            log.info("Crawling [%d/%d]: %s", count, MAX_PAGES, url[:90])
            soup = self._fetch(url)
            if soup is None:
                continue

            self._save(url, soup)

            new_links = extract_links(url, soup)
            for link in new_links:
                if link not in self.visited:
                    self.queue.append(link)

            time.sleep(DELAY_SEC)

        pdf_list_path = self.output_dir / "_pdf_urls.json"
        unique_pdfs = list(set(self.pdf_urls))
        pdf_list_path.write_text(json.dumps(unique_pdfs, indent=2), encoding="utf-8")

        log.info("Crawl complete. Visited: %d | Saved: %d | PDFs found: %d",
                 count, self.saved, len(unique_pdfs))
