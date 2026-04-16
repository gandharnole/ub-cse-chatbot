"""
run_scraper.py
Entry point for the full data collection pipeline.

Usage:
    python scraper/run_scraper.py            # full pipeline
    python scraper/run_scraper.py --crawl    # crawl only
    python scraper/run_scraper.py --pdfs     # PDF extraction only
    python scraper/run_scraper.py --clean    # cleaning only
"""

import argparse
import logging
import sys
from pathlib import Path

# Make sure project root is on sys.path when running from any directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scraper.crawler       import Crawler, SEED_URLS
from scraper.pdf_extractor import run_pdf_extraction
from scraper.cleaner       import run_cleaning

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data/scraper.log", mode="w"),
    ]
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="UB CSE scraper pipeline")
    parser.add_argument("--crawl", action="store_true", help="Run web crawler only")
    parser.add_argument("--pdfs",  action="store_true", help="Run PDF extractor only")
    parser.add_argument("--clean", action="store_true", help="Run text cleaner only")
    args = parser.parse_args()

    run_all = not any([args.crawl, args.pdfs, args.clean])

    # Ensure data dirs exist
    Path("data/raw").mkdir(parents=True, exist_ok=True)

    if run_all or args.crawl:
        log.info("=" * 50)
        log.info("STEP 1 — Web crawl")
        log.info("=" * 50)
        crawler = Crawler()
        crawler.run(SEED_URLS)

    if run_all or args.pdfs:
        log.info("=" * 50)
        log.info("STEP 2 — PDF extraction")
        log.info("=" * 50)
        run_pdf_extraction()

    if run_all or args.clean:
        log.info("=" * 50)
        log.info("STEP 3 — Text cleaning")
        log.info("=" * 50)
        run_cleaning()

    log.info("Pipeline finished. Raw data → data/raw/")


if __name__ == "__main__":
    main()
