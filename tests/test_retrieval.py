"""Tests for retrieval. Owner: Minh. Offline: no network, no API key."""
import pytest

from seminar_critique.retrieval.chunker import (
    CHUNK_WORDS, OVERLAP_WORDS, chunk_paper, split_into_chunks,
)
from seminar_critique.retrieval.ingest import parse_papers_md, sections_from_text, title_on_first_page

SAMPLE = """
Large Language Models Cannot Self-Correct Reasoning Yet
Abstract
We study self-correction. """ + "word " * 30 + """
1 Introduction
""" + "intro " * 40 + """
2 Method
2.1 Intrinsic self-correction
""" + "method " * 40 + """
References
Some Author. Should not appear.
"""


def test_split_respects_size_and_overlap():
    words = [f"w{i}" for i in range(1000)]
    chunks = split_into_chunks(" ".join(words))
    assert all(len(c.split()) <= CHUNK_WORDS for c in chunks)
    first, second = chunks[0].split(), chunks[1].split()
    assert first[-OVERLAP_WORDS:] == second[:OVERLAP_WORDS]
    assert chunks[-1].split()[-1] == "w999"  # nothing is lost at the end


def test_sections_found_and_references_dropped():
    sections = dict(sections_from_text(SAMPLE))
    assert "Abstract" in sections
    assert "Section 1 Introduction" in sections
    assert "Section 2.1 Intrinsic self-correction" in sections
    assert "Should not appear" not in " ".join(sections.values())


def test_chunk_metadata_is_complete_and_ids_unique():
    paper = {"id": "huang2024", "topic": "self-improvement-reflection",
             "cite": "Huang et al. 2024", "title": "x", "arxiv": "2310.01798"}
    chunks = chunk_paper(paper, sections_from_text(SAMPLE))
    assert chunks
    assert len({c["id"] for c in chunks}) == len(chunks)
    for c in chunks:
        assert c["paper"] == "Huang et al. 2024" and c["section"] and c["text"]


def test_papers_table_parses():
    rows = parse_papers_md()
    assert len(rows) >= 4
    assert {"id", "topic", "cite", "title", "arxiv"} <= set(rows[0])
    assert len({r["id"] for r in rows}) == len(rows)


def test_pdf_roundtrip(tmp_path):
    pymupdf = pytest.importorskip("pymupdf")
    from seminar_critique.retrieval.ingest import pdf_to_text
    doc = pymupdf.open()
    page = doc.new_page()
    y = 72
    for line in ["My Test Paper Title", "Abstract", "short abstract text here", "1 Introduction", "body of the intro", "References"]:
        page.insert_text((72, y), line)
        y += 20
    path = tmp_path / "t.pdf"
    doc.save(path)
    text, first = pdf_to_text(path)
    assert title_on_first_page("My Test Paper Title", first)
    assert "Section 1 Introduction" in dict(sections_from_text(text))


NOISY = """
Large Language Models Cannot Self-Correct Reasoning Yet
1 Work done at Google. 2 UC Berkeley
Abstract
""" + "abstract " * 30 + """
1 INTRODUCTION
""" + "intro " * 40 + """
2 BACKGROUND AND RELATED WORK
""" + "background " * 40 + """
4 Llama 70 12
3 Method
""" + "method " * 40 + """
3.1 Setup
""" + "setup " * 40 + """
References
"""


def test_footnotes_and_table_rows_are_not_headings():
    names = [n for n, _ in sections_from_text(NOISY)]
    assert names == [
        "Abstract", "Section 1 INTRODUCTION", "Section 2 BACKGROUND AND RELATED WORK",
        "Section 3 Method", "Section 3.1 Setup",
    ]


def test_running_header_and_page_numbers_removed():
    from seminar_critique.retrieval.ingest import strip_page_noise
    pages = [
        "Published as a conference paper at ICLR 2024\ntext one\n1",
        "Published as a conference paper at ICLR 2024\n2 Published as a conference paper at ICLR 2024\ntext two\n2",
        "Published as a conference paper at ICLR 2024\ntext three\n3",
    ]
    out = " ".join(strip_page_noise(pages))
    assert "Published as a conference" not in out
    assert "text one" in out and "text three" in out
    assert not any(line.strip() in {"1", "2", "3"} for line in out.split())
