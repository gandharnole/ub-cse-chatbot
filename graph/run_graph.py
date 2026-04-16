"""
run_graph.py — Build and save the UB CSE knowledge graph.

Usage:
    python -m graph.run_graph
    python -m graph.run_graph --stats
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.entity_extractor import extract_all
from graph.kg_builder       import build_graph, save_graph, load_graph

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats", action="store_true", help="Print graph stats only")
    args = parser.parse_args()

    kg_path = Path("graph/kg_store.json")

    if args.stats:
        if not kg_path.exists():
            log.error("No graph found. Run without --stats first.")
            return
        G = load_graph(kg_path)
        node_types = {}
        for n in G.nodes:
            t = G.nodes[n].get("type", "unknown")
            node_types[t] = node_types.get(t, 0) + 1
        log.info("Graph stats:")
        log.info("  Total nodes : %d", G.number_of_nodes())
        log.info("  Total edges : %d", G.number_of_edges())
        for t, count in sorted(node_types.items()):
            log.info("  %-20s: %d", t, count)
        return

    log.info("Step 1 — Extracting entities from raw data...")
    entities = extract_all()

    log.info("Step 2 — Building knowledge graph...")
    G = build_graph(entities)

    log.info("Step 3 — Saving graph...")
    save_graph(G, kg_path)

    # Also save flat entity lists for easy inspection
    entities_path = Path("graph/entities.json")
    entities_path.write_text(
        json.dumps(entities, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    log.info("Entities saved to %s", entities_path)
    log.info("Knowledge graph complete.")


if __name__ == "__main__":
    main()
