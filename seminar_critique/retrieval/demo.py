"""Show what retrieve() returns, with and without the graph. Owner: Minh.

Use it to pick examples for the report (Part 4.4: one correct, one wrong):
    python -m seminar_critique.retrieval.demo "your query" self-improvement-reflection
    python -m seminar_critique.retrieval.demo "your query" tool-use-mcp -k 5 --width 300
"""
from __future__ import annotations

import argparse
import sys

from .retriever import retrieve_with


def show(title: str, chunks, width: int) -> None:
    print(f"\n== {title}")
    for i, c in enumerate(chunks, 1):
        snippet = " ".join(c.text.split())[:width]
        print(f"{i}. {c.paper} | {c.section} | {c.id} | score={c.score}")
        print(f"   {snippet}...")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles choke on quotes
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("topic")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--width", type=int, default=160, help="characters of text to show per chunk")
    args = ap.parse_args()
    print(f"query: {args.query}\ntopic: {args.topic}  k={args.k}")
    show("BM25 only", retrieve_with(args.query, args.topic, args.k, expand=False), args.width)
    show("BM25 + graph", retrieve_with(args.query, args.topic, args.k, expand=True), args.width)


if __name__ == "__main__":
    main()
