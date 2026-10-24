"""Write the retrieval numbers needed by the report (Part 4). Owner: Minh.

    python -m seminar_critique.retrieval.stats
Output: seminar_critique/retrieval/stats.json (commit it)
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from . import retriever
from .chunker import CHUNK_WORDS, OVERLAP_WORDS
from .graph import SIM_MIN, SIM_TOP, graph_stats

OUT = Path(__file__).with_name("stats.json")


def compute() -> dict:
    ix = retriever.get_index()
    words = [c["n_words"] for c in ix.chunks]
    per_topic = Counter(c["topic"] for c in ix.chunks)
    papers_per_topic = {t: len({ix.chunks[i]["paper_id"] for i in idxs}) for t, idxs in ix.by_topic.items()}
    return {
        "papers": len({c["paper_id"] for c in ix.chunks}),
        "papers_per_topic": papers_per_topic,
        "chunks": len(ix.chunks),
        "chunks_per_topic": dict(per_topic),
        "chunk_size_words": CHUNK_WORDS,
        "chunk_overlap_words": OVERLAP_WORDS,
        "avg_words_per_chunk": round(sum(words) / len(words), 1),
        "indexing": f"BM25 (k1={retriever.BM25_K1}, b={retriever.BM25_B}), lexical, no embedding model",
        "graph": graph_stats(ix.graph),
        "graph_edge_rules": {
            "adjacent": "consecutive chunks of one paper",
            "similar": f"top {SIM_TOP} TF-IDF cosine neighbours in other papers of the same topic, cosine >= {SIM_MIN}",
            "cites": "paper text mentions 'Surname et al., year' of another paper in the corpus",
        },
        "top_k_default": retriever.DEFAULT_K,
        "graph_expansion": {"propagation_weight": retriever.PROPAGATION, "cite_bonus": retriever.CITE_BONUS},
        "max_chunks_per_paper_in_top_k": retriever.MAX_PER_PAPER,
    }


if __name__ == "__main__":
    stats = compute()
    OUT.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))
