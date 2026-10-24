"""Tokenizer shared by the graph (TF-IDF similarity) and the retriever (BM25). Owner: Minh."""
from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an and are as at be been but by can could did do does for from had has have how if in into is it its
may might more most not of on or our out over should so some such than that the their them then there
these they this those to too under up use used using was we were what when where which while who whom why
will with would you your about also both each other than via per etc eg ie
""".split())

_TOKEN = re.compile(r"[a-z0-9]+")


def _stem(tok: str) -> str:
    """Very light stemming so 'corrections' and 'correction' match."""
    for suffix in ("ing", "es", "ed", "s"):
        if tok.endswith(suffix) and len(tok) - len(suffix) >= 4:
            return tok[: -len(suffix)]
    return tok


def tokenize(text: str) -> list[str]:
    return [
        _stem(t) for t in _TOKEN.findall(text.lower())
        if len(t) >= 2 and t not in STOPWORDS
    ]
