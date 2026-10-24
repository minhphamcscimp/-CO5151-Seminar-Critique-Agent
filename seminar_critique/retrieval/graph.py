"""Paper graph for GraphRAG. Owner: Minh.

Nodes
    paper:<id>   one per paper
    <chunk id>   one per chunk
Edges (attribute "kind")
    part_of    chunk  - its paper
    adjacent   chunk  - next chunk of the same paper (reading order)
    similar    chunk  - its most similar chunks in OTHER papers of the same topic (TF-IDF cosine)
    cites      paper  - paper, when one paper's text mentions the other ("Huang et al., 2024")

The "cites" edges come from a text match, so they can include false positives
(two different first authors with the same surname). Say this in the report.
"""
from __future__ import annotations

import math
import re
from collections import Counter

import networkx as nx

from .textutil import tokenize

SIM_TOP = 3        # similar-edges kept per chunk
SIM_MIN = 0.15     # minimum cosine for a similar-edge


def _tfidf_vectors(token_lists: list[list[str]]) -> list[dict[str, float]]:
    n = len(token_lists)
    df = Counter(t for toks in token_lists for t in set(toks))
    vectors = []
    for toks in token_lists:
        tf = Counter(toks)
        vec = {t: (1 + math.log(c)) * math.log(1 + n / df[t]) for t, c in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        vectors.append({t: v / norm for t, v in vec.items()})
    return vectors


def _cosine(a: dict[str, float], b: dict[str, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(t, 0.0) for t, v in a.items())


def _citation_patterns(papers: dict[str, dict]) -> dict[str, re.Pattern]:
    """'Huang et al. 2024' -> regex for 'Huang et al., 2023/2024/2025'."""
    patterns = {}
    for pid, p in papers.items():
        m = re.match(r"(\w+) et al\.? (\d{4})", p["cite"])
        if m:
            surname, year = m.group(1), int(m.group(2))
            years = "|".join(str(y) for y in (year - 1, year, year + 1))
            patterns[pid] = re.compile(rf"\b{surname}\s+et\s+al\.?,?\s*\(?(?:{years})")
    return patterns


def build_graph(chunks: list[dict], token_lists: list[list[str]] | None = None) -> nx.Graph:
    if token_lists is None:
        token_lists = [tokenize(c["text"]) for c in chunks]
    G = nx.Graph()
    papers: dict[str, dict] = {}
    for c in chunks:
        papers.setdefault(c["paper_id"], {"cite": c["paper"], "topic": c["topic"]})
    for pid, p in papers.items():
        G.add_node(f"paper:{pid}", type="paper", cite=p["cite"], topic=p["topic"])

    last_in_paper: dict[str, str] = {}
    for c in chunks:
        G.add_node(c["id"], type="chunk", paper_id=c["paper_id"], topic=c["topic"], section=c["section"])
        G.add_edge(c["id"], f"paper:{c['paper_id']}", kind="part_of", weight=1.0)
        prev = last_in_paper.get(c["paper_id"])
        if prev:
            G.add_edge(prev, c["id"], kind="adjacent", weight=1.0)
        last_in_paper[c["paper_id"]] = c["id"]

    vectors = _tfidf_vectors(token_lists)
    for i, ci in enumerate(chunks):
        sims = []
        for j, cj in enumerate(chunks):
            if i == j or ci["paper_id"] == cj["paper_id"] or ci["topic"] != cj["topic"]:
                continue
            cos = _cosine(vectors[i], vectors[j])
            if cos >= SIM_MIN:
                sims.append((cos, j))
        for cos, j in sorted(sims, reverse=True)[:SIM_TOP]:
            if not G.has_edge(ci["id"], chunks[j]["id"]):
                G.add_edge(ci["id"], chunks[j]["id"], kind="similar", weight=round(cos, 4))

    patterns = _citation_patterns(papers)
    mentions: Counter = Counter()
    for c in chunks:
        for target, pat in patterns.items():
            if target != c["paper_id"] and pat.search(c["text"]):
                mentions[tuple(sorted((c["paper_id"], target)))] += 1
    for (a, b), n in mentions.items():
        G.add_edge(f"paper:{a}", f"paper:{b}", kind="cites", weight=float(n))
    return G


def graph_stats(G: nx.Graph) -> dict:
    kinds = Counter(d["kind"] for _, _, d in G.edges(data=True))
    types = Counter(d["type"] for _, d in G.nodes(data=True))
    return {
        "nodes": G.number_of_nodes(),
        "nodes_by_type": dict(types),
        "edges": G.number_of_edges(),
        "edges_by_kind": dict(kinds),
    }
