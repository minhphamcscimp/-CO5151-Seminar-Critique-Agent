"""GraphRAG retrieval. Owner: Minh.

CONTRACT (do not change the signature without a "[contract]" PR):
    retrieve(query, topic, k) -> list[Chunk]

How it works (all numbers are logged by stats.py for the report, Part 4):
    1. BM25 over the chunks of the topic (lexical, no embedding model).
    2. Graph expansion: a chunk gets a bonus from its graph neighbours that are
       strong BM25 hits (adjacent chunk, similar chunk in another paper), and a
       small bonus when its paper is linked by a "cites" edge to the best hit's paper.
    3. Return the top-k by final score.

`retrieve_with(..., expand=False)` gives plain BM25, for the ablation "with vs without graph".
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from ..schemas import Chunk
from .chunker import read_jsonl
from .graph import build_graph
from .textutil import tokenize

CHUNKS_JSONL = Path(__file__).resolve().parents[2] / "data" / "chunks.jsonl"

BM25_K1 = 1.5
BM25_B = 0.75
PROPAGATION = 0.5   # weight of a neighbour's score
CITE_BONUS = 0.1    # bonus for papers linked to the best hit's paper
DEFAULT_K = 5


class _Index:
    def __init__(self, chunks: list[dict]):
        self.chunks = chunks
        self.tokens = [tokenize(c["text"]) for c in chunks]
        self.pos = {c["id"]: i for i, c in enumerate(chunks)}
        n = len(chunks)
        self.avgdl = sum(len(t) for t in self.tokens) / max(n, 1)
        df = Counter(t for toks in self.tokens for t in set(toks))
        self.idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.tf = [Counter(toks) for toks in self.tokens]
        self.by_topic: dict[str, list[int]] = defaultdict(list)
        for i, c in enumerate(chunks):
            self.by_topic[c["topic"]].append(i)
        self.graph = build_graph(chunks, self.tokens)
        # chunk-level neighbours and paper-level links, read from the graph
        self.neighbors: dict[int, list[tuple[int, float]]] = defaultdict(list)
        self.paper_links: dict[str, set[str]] = defaultdict(set)
        for u, v, d in self.graph.edges(data=True):
            if d["kind"] in ("adjacent", "similar"):
                i, j = self.pos[u], self.pos[v]
                self.neighbors[i].append((j, d["weight"]))
                self.neighbors[j].append((i, d["weight"]))
            elif d["kind"] == "cites":
                a, b = u.removeprefix("paper:"), v.removeprefix("paper:")
                self.paper_links[a].add(b)
                self.paper_links[b].add(a)

    def bm25(self, query_tokens: list[str], idxs: list[int]) -> dict[int, float]:
        scores: dict[int, float] = {}
        for i in idxs:
            dl, tf, s = len(self.tokens[i]), self.tf[i], 0.0
            for t in set(query_tokens):
                f = tf.get(t, 0)
                if f:
                    s += self.idf[t] * f * (BM25_K1 + 1) / (f + BM25_K1 * (1 - BM25_B + BM25_B * dl / self.avgdl))
            if s > 0:
                scores[i] = s
        return scores


@lru_cache(maxsize=1)
def get_index(path: Path = CHUNKS_JSONL) -> _Index:
    return _Index(read_jsonl(path))


def _to_chunk(c: dict, score: float) -> Chunk:
    return Chunk(id=c["id"], paper=c["paper"], section=c["section"], text=c["text"], score=round(score, 4))


def retrieve_with(query: str, topic: str, k: int = DEFAULT_K, expand: bool = True) -> list[Chunk]:
    ix = get_index()
    idxs = ix.by_topic.get(topic) or list(range(len(ix.chunks)))  # unknown topic: search everything
    scores = ix.bm25(tokenize(query), idxs)
    if not scores:  # nothing matches: keep the contract and return the first chunks
        return [_to_chunk(ix.chunks[i], 0.0) for i in idxs[:k]]

    ranked = sorted(scores, key=scores.get, reverse=True)
    top = scores[ranked[0]]
    norm = {i: scores[i] / top for i in ranked}
    if not expand:
        return [_to_chunk(ix.chunks[i], scores[i]) for i in ranked[:k]]

    allowed = set(idxs)
    seeds = ranked[:k]
    final = {i: norm[i] for i in seeds}
    for s in seeds:
        for j, w in ix.neighbors[s]:
            if j in allowed:
                final.setdefault(j, norm.get(j, 0.0))
    for j in list(final):
        bonus = max((PROPAGATION * w * norm[s] for s in seeds for n, w in ix.neighbors[s] if n == j), default=0.0)
        final[j] = norm.get(j, 0.0) + bonus
    best_paper = ix.chunks[seeds[0]]["paper_id"]
    for j in final:
        if ix.chunks[j]["paper_id"] in ix.paper_links[best_paper]:
            final[j] += CITE_BONUS
    out = sorted(final, key=final.get, reverse=True)[:k]
    return [_to_chunk(ix.chunks[i], final[i]) for i in out]


def retrieve(query: str, topic: str, k: int = DEFAULT_K) -> list[Chunk]:
    """Top-k chunks of the topic for the query (BM25 + graph expansion)."""
    return retrieve_with(query, topic, k, expand=True)
