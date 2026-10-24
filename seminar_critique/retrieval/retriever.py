"""GraphRAG retrieval. Owner: Minh.

CONTRACT (do not change the signature without a "[contract]" PR):
    retrieve(query, topic, k) -> list[Chunk]

Planned files in this folder (create them as you go):
    ingest.py     read papers (PDF or text) for 1 to 2 topics first
    chunker.py    split by section, record chunk size
    graph.py      build the paper graph (nodes = papers/chunks, edges = citations or shared topic)
    retriever.py  this file: vector search, then expand along graph edges

The report (Part 4) needs these numbers, so log them: papers ingested,
chunks created, graph nodes and edges, chunk size, embedding model, top-k.
"""
from __future__ import annotations

from ..schemas import Chunk


def retrieve(query: str, topic: str, k: int = 5) -> list[Chunk]:
    """STUB: returns one fixed chunk so other modules can run end-to-end.

    TODO (Minh): replace with real retrieval. Keep the return type.
    """
    return [
        Chunk(
            id="stub-0",
            paper="Huang et al. 2024",
            section="Section 3.1",
            text="(stub) Intrinsic self-correction without external feedback does not improve reasoning.",
            score=1.0,
        )
    ][:k]
