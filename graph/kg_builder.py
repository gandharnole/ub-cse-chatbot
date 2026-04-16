"""
kg_builder.py
Builds a knowledge graph from extracted entities using networkx.
Stores the graph as kg_store.json for fast lookup at query time.
"""

import json
import logging
from pathlib import Path

import networkx as nx

from graph.entity_extractor import extract_all

log = logging.getLogger(__name__)

KG_PATH = Path("graph/kg_store.json")


def build_graph(entities: dict) -> nx.DiGraph:
    G = nx.DiGraph()

    # ── Add course nodes ──────────────────────────────────────────────────────
    for course in entities["courses"]:
        code = course["code"]
        G.add_node(code, type="course", **{
            k: v for k, v in course.items() if k != "code"
        })

    # ── Add prerequisite edges ────────────────────────────────────────────────
    for course in entities["courses"]:
        for prereq in course.get("prereqs", []):
            if prereq in G:
                G.add_edge(prereq, course["code"], relation="prereq_of")

    # ── Add faculty nodes and edges ───────────────────────────────────────────
    for fac in entities["faculty"]:
        name = fac["name"]
        G.add_node(name, type="faculty", **{
            k: v for k, v in fac.items() if k != "name"
        })

        # Faculty → Course (teaches)
        for course_code in fac.get("courses_taught", []):
            if course_code in G:
                G.add_edge(name, course_code, relation="teaches")

        # Faculty → Research area
        for area in fac.get("research_areas", []):
            area_node = f"area::{area}"
            if not G.has_node(area_node):
                G.add_node(area_node, type="research_area", name=area)
            G.add_edge(name, area_node, relation="researches")

    # ── Add lab nodes and edges ───────────────────────────────────────────────
    for lab in entities["labs"]:
        lab_node = f"lab::{lab['name'][:60]}"
        G.add_node(lab_node, type="lab", name=lab["name"], source=lab["source"])

        for area in lab.get("research_areas", []):
            area_node = f"area::{area}"
            if not G.has_node(area_node):
                G.add_node(area_node, type="research_area", name=area)
            G.add_edge(lab_node, area_node, relation="researches")

    log.info(
        "Graph built. Nodes: %d | Edges: %d",
        G.number_of_nodes(), G.number_of_edges()
    )
    return G


def save_graph(G: nx.DiGraph, path: Path = KG_PATH) -> None:
    """Serialize graph to JSON for fast loading at query time."""
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "nodes": [
            {"id": n, **G.nodes[n]}
            for n in G.nodes
        ],
        "edges": [
            {"source": u, "target": v, **G.edges[u, v]}
            for u, v in G.edges
        ],
    }
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Graph saved to %s", path)


def load_graph(path: Path = KG_PATH) -> nx.DiGraph:
    """Load graph from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    G = nx.DiGraph()
    for node in data["nodes"]:
        nid = node.pop("id")
        G.add_node(nid, **node)
    for edge in data["edges"]:
        G.add_edge(edge["source"], edge["target"], relation=edge.get("relation", ""))
    return G


# ── Query helpers used by the API ─────────────────────────────────────────────

def get_course_info(G: nx.DiGraph, course_code: str) -> dict:
    """Return all info about a course including faculty and prereqs."""
    code = course_code.upper().replace(" ", " ")  # normalise
    if code not in G:
        # Try fuzzy: CSE574 → CSE 574
        import re
        m = re.match(r"CSE(\d+)", code, re.I)
        if m:
            code = f"CSE {m.group(1)}"

    if code not in G:
        return {}

    node = dict(G.nodes[code])

    # Who teaches this course?
    teachers = [
        u for u, v, d in G.in_edges(code, data=True)
        if d.get("relation") == "teaches"
    ]
    node["taught_by"] = teachers

    # What does it unlock (courses that list it as prereq)?
    unlocks = [
        v for u, v, d in G.out_edges(code, data=True)
        if d.get("relation") == "prereq_of"
    ]
    node["unlocks"] = unlocks

    return node


def get_faculty_info(G: nx.DiGraph, name: str) -> dict:
    """Return faculty info including courses and related labs."""
    # Try exact, then partial match
    if name not in G:
        matches = [n for n in G.nodes if name.lower() in n.lower()
                   and G.nodes[n].get("type") == "faculty"]
        if not matches:
            return {}
        name = matches[0]

    node = dict(G.nodes[name])

    courses = [
        v for u, v, d in G.out_edges(name, data=True)
        if d.get("relation") == "teaches"
    ]
    areas = [
        G.nodes[v]["name"] for u, v, d in G.out_edges(name, data=True)
        if d.get("relation") == "researches" and G.nodes[v].get("type") == "research_area"
    ]

    # Suggest related labs based on shared research areas
    related_labs = []
    for area_node in [v for _, v, d in G.out_edges(name, data=True)
                      if d.get("relation") == "researches"]:
        for lab_node, _, d2 in G.in_edges(area_node, data=True):
            if G.nodes[lab_node].get("type") == "lab":
                related_labs.append(G.nodes[lab_node]["name"])

    node["courses_taught"]  = courses
    node["research_areas"]  = areas
    node["related_labs"]    = list(set(related_labs))
    return node


def suggest_related(G: nx.DiGraph, course_code: str) -> dict:
    """BONUS: Given a course, suggest related faculty and labs."""
    info  = get_course_info(G, course_code)
    result = {"course": course_code, "faculty": [], "labs": []}

    for teacher in info.get("taught_by", []):
        fac_info = get_faculty_info(G, teacher)
        result["faculty"].append({
            "name":  teacher,
            "areas": fac_info.get("research_areas", []),
            "labs":  fac_info.get("related_labs", []),
        })
        result["labs"].extend(fac_info.get("related_labs", []))

    result["labs"] = list(set(result["labs"]))
    return result
