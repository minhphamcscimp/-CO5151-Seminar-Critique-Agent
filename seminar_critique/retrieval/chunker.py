"""Split paper sections into chunks. Owner: Minh.

Chunk size is a REPORT NUMBER (Part 4): keep CHUNK_WORDS and OVERLAP_WORDS here
as the single source of truth, and quote them in the report.
"""
from __future__ import annotations

import json
from pathlib import Path

CHUNK_WORDS = 250      # max words per chunk
OVERLAP_WORDS = 40     # words shared by two neighbouring chunks of the same section
MIN_WORDS = 20         # drop chunks shorter than this (page numbers, noise)


def split_into_chunks(
    text: str, chunk_words: int = CHUNK_WORDS, overlap: int = OVERLAP_WORDS
) -> list[str]:
    """Cut text into windows of `chunk_words` words, neighbours share `overlap` words."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_words:
        return [" ".join(words)]
    step = chunk_words - overlap
    pieces = []
    for start in range(0, len(words), step):
        pieces.append(" ".join(words[start:start + chunk_words]))
        if start + chunk_words >= len(words):
            break
    return pieces


def chunk_paper(paper: dict, sections: list[tuple[str, str]]) -> list[dict]:
    """paper: row of data/papers.md (id, topic, cite, title, arxiv).
    sections: [(section_label, section_text), ...]
    Returns one dict per chunk, ready for JSONL.
    """
    chunks: list[dict] = []
    for label, text in sections:
        for piece in split_into_chunks(text):
            n_words = len(piece.split())
            if n_words < MIN_WORDS:
                continue
            chunks.append({
                "id": f"{paper['id']}-{len(chunks):03d}",
                "paper_id": paper["id"],
                "paper": paper["cite"],
                "topic": paper["topic"],
                "section": label,
                "text": piece,
                "n_words": n_words,
            })
    return chunks


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
