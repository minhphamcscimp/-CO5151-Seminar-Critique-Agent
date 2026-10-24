"""Read papers (PDF or text) and split them into sections. Owner: Minh.

Run from the repo root:
    python -m seminar_critique.retrieval.ingest
    python -m seminar_critique.retrieval.ingest --topic self-improvement-reflection

What it does:
    1. read data/papers.md
    2. download each PDF from arXiv to data/raw/<id>.pdf (not committed)
       (or use data/raw/<id>.txt if you put a plain text file there)
    3. split the text into sections by heading (heuristic, check the printout)
    4. cut sections into chunks and write data/chunks.jsonl (committed)
"""
from __future__ import annotations

import argparse
import re
import time
import urllib.request
from pathlib import Path

from .chunker import CHUNK_WORDS, OVERLAP_WORDS, chunk_paper, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
PAPERS_MD = DATA / "papers.md"
CHUNKS_JSONL = DATA / "chunks.jsonl"

# "3.1 Method" on one line
NUM_TITLE = re.compile(r"^(\d{1,2}(?:\.[1-9]){0,2})\.?\s*([A-Z][^.!?]{2,80})$")
# "3.1" alone on a line, title on the next line
NUM_ONLY = re.compile(r"^(\d{1,2}(?:\.[1-9]){0,2})\.?$")
NAMED = {
    "abstract", "introduction", "related work", "background", "conclusion",
    "conclusions", "limitations", "discussion", "preliminaries",
}
STOP = {"references", "bibliography", "acknowledgments", "acknowledgements"}

# pdf_to_text puts this prefix on lines printed in a bold font. Headings are bold in almost every
# paper template, table rows and figure labels are not, so bold is the best clue we have.
BOLD_MARK = "\u00a7\u00a7 "
BOLD_HINTS = ("bold", "medi", "black", "heavy", "semib", "-bd", "bx")


def parse_papers_md(path: Path = PAPERS_MD) -> list[dict]:
    """Read the markdown table in data/papers.md."""
    cols = ["id", "topic", "cite", "title", "arxiv"]
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) != len(cols) or cells[0] in ("id", "") or set(cells[0]) <= {"-", ":"}:
            continue
        rows.append(dict(zip(cols, cells)))
    return rows


_SMALL = frozenset("and of for the to in on with via a an by from as at or vs is are not do".split())
_BAD_START = frozenset("we our in this it for these each".split())


def _heading_like(title: str) -> bool:
    """Headings are short; footnotes and body lines are sentences ("We discard all examples ...")."""
    words = title.split()
    if not 1 <= len(words) <= 10 or words[0].lower() in _BAD_START or any(ch in title for ch in "\\$^~|<>@") \
            or any(len(w) > 18 for w in words):
        return False
    return len([w for w in words if w[0].islower() and w not in _SMALL]) <= 5


def _looks_like_title(t: str) -> bool:
    return (3 <= len(t) <= 80 and t[0].isupper() and len(t.split()) <= 10
            and not t.endswith((".", ":", ",")) and _heading_like(t))


def _parse_num(num: str) -> tuple[int, ...]:
    return tuple(int(x) for x in num.split("."))


def _numbering_ok(nums: tuple[int, ...], last: tuple[int, ...]) -> bool:
    """A real heading number goes forward: 2 -> 2.1 -> 2.2 -> 3. Rejects page numbers,
    footnotes ("1 Work done at ...") and table rows that look like headings."""
    max_jump = 2 if last[0] > 0 else 1
    return nums > last and nums[0] <= last[0] + max_jump


def _split_sections(text: str, use_bold: bool) -> list[tuple[str, str]]:
    """Split raw paper text into [(section_label, text)]. Stops at References."""
    text = re.sub(r"(\w)-\n([a-z])", r"\1\2", text)  # join words split by a line break
    has_marks = use_bold and BOLD_MARK in text   # PDF text with font info: headings must be bold
    raw = [ln.strip() for ln in text.splitlines() if ln.strip()]
    lines = [ln.removeprefix(BOLD_MARK.strip()).strip() for ln in raw]
    bold = [(not has_marks) or ln.startswith(BOLD_MARK.strip()) for ln in raw]
    sections: list[tuple[str, str]] = []
    label, buf = "Front matter", []
    last: tuple[int, ...] = (0,)   # last accepted heading number
    started = False                # True after "Abstract" or "Introduction" is seen

    def flush() -> None:
        if buf:
            sections.append((label, " ".join(buf)))

    def numbered(num: str, title: str) -> str | None:
        nonlocal last, started
        nums = _parse_num(num)
        if any(tok.isdigit() for tok in title.split()) or not _heading_like(title):
            return None  # table row such as "4 Llama 70 12", or a footnote sentence
        if title.lower().startswith("introduction"):
            last, started = nums, True
        elif not (started and _numbering_ok(nums, last)):
            return None
        else:
            last = nums
        return f"Section {num} {title}"

    i = 0
    while i < len(lines):
        s = lines[i]
        low = s.lower().strip(":")
        new_label = None
        if low in STOP:
            break
        if bold[i] and low in NAMED and len(s) < 30:
            new_label = s.strip(":").title()
            if low in ("abstract", "introduction"):
                started = True
            if low == "introduction" and last == (0,):
                last = (1,)
        elif bold[i] and (m := NUM_TITLE.match(s)):
            new_label = numbered(m.group(1), m.group(2))
        elif (m := NUM_ONLY.match(s)) and i + 1 < len(lines) and bold[i + 1] and _looks_like_title(lines[i + 1]):
            new_label = numbered(m.group(1), lines[i + 1])
            if new_label:
                i += 1
        if new_label:
            flush()
            label, buf = new_label, []
        else:
            buf.append(s)
        i += 1
    flush()
    # Title and authors come before the "Abstract" heading: drop them when the heading exists,
    # otherwise treat the front matter as the abstract.
    if any(name == "Abstract" for name, _ in sections):
        sections = [(n, t) for n, t in sections if n != "Front matter"]
    else:
        sections = [("Abstract" if n == "Front matter" else n, t) for n, t in sections]
    return sections


def sections_from_text(text: str) -> list[tuple[str, str]]:
    """Split a paper into sections.

    With font info (PDF) we try two readings: headings must be bold, or any heading-like line.
    We keep the one that found more numbered sections; on a tie we keep the bold one (fewer false hits).
    """
    plain = _split_sections(text, use_bold=False)
    if BOLD_MARK not in text:
        return plain
    bold = _split_sections(text, use_bold=True)

    def score(result: list[tuple[str, str]]) -> int:
        return sum(1 for name, _ in result if name.startswith("Section "))

    return bold if score(bold) >= score(plain) else plain


def drop_garbled_lines(text: str) -> tuple[str, int]:
    """Some PDFs have broken font encoding, so extraction gives letters like 'XHVWLRQ $V'.
    Drop long lines that contain no common English word. Headings are kept."""
    common = {"the", "of", "and", "to", "in", "a", "is", "we", "for", "that", "with", "on", "as", "by", "this"}
    kept, dropped = [], 0
    for line in text.splitlines():
        plain = line.removeprefix(BOLD_MARK.strip()).strip()
        if plain.lower().strip(":") in STOP:
            break  # the reference list: not used, not counted
        words = set(re.findall(r"[a-z]+", plain.lower()))
        heading = (m := NUM_TITLE.match(plain)) is not None and " " in m.group(2)
        if len(plain) >= 30 and not (words & common) and not line.startswith(BOLD_MARK.strip()) and not heading:
            dropped += 1
            continue
        kept.append(line)
    return "\n".join(kept), dropped


def strip_page_noise(pages: list[str]) -> list[str]:
    """Remove page numbers and running headers/footers (lines repeated on many pages)."""
    split = [[ln.strip() for ln in pg.splitlines() if ln.strip()] for pg in pages]
    counts: dict[str, int] = {}
    for lines in split:
        for key in {re.sub(r"\d+", "", ln).lower().strip() for ln in lines}:
            if key:
                counts[key] = counts.get(key, 0) + 1
    threshold = max(3, int(0.4 * len(pages)))
    cleaned = []
    for lines in split:
        lines = list(lines)
        for idx in (0, -1):  # a bare number at the top or bottom of a page is a page number
            if lines and re.fullmatch(r"\d{1,3}", lines[idx].removeprefix(BOLD_MARK.strip()).strip()):
                lines.pop(idx)
        kept = []
        for ln in lines:
            key = re.sub(r"\d+", "", ln).lower().strip()
            if key and len(ln) < 100 and counts.get(key, 0) >= threshold:
                continue
            kept.append(ln)
        cleaned.append("\n".join(kept))
    return cleaned


def download_pdf(arxiv_id: str, dest: Path) -> bool:
    """Download once. Returns True if a download happened."""
    if dest.exists() and dest.stat().st_size > 10_000:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        f"https://arxiv.org/pdf/{arxiv_id}",
        headers={"User-Agent": "Mozilla/5.0 (university course project)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        dest.write_bytes(resp.read())
    return True


def _line_text(line: dict) -> str:
    """Join the spans of a line, adding a space where the PDF leaves a gap ("1" + "INTRODUCTION")."""
    out, prev = "", None
    for sp in line["spans"]:
        t = sp["text"]
        if prev is not None and sp["bbox"][0] - prev["bbox"][2] > 0.15 * sp["size"] \
                and not out.endswith(" ") and not t.startswith(" "):
            out += " "
        out += t
        prev = sp
    return re.sub(r"\s+", " ", out).strip()


def _line_is_bold(line: dict) -> bool:
    spans = [sp for sp in line["spans"] if sp["text"].strip()]
    return bool(spans) and all(
        (sp["flags"] & 16) or any(h in sp["font"].lower() for h in BOLD_HINTS) for sp in spans
    )


def pdf_to_text(pdf_path: Path) -> tuple[str, str]:
    """Return (full_text, first_page_text). Bold lines get the BOLD_MARK prefix."""
    import pymupdf  # pip install pymupdf

    pages = []
    with pymupdf.open(pdf_path) as doc:
        first = doc[0].get_text("text") if len(doc) else ""
        for page in doc:
            out = []
            for block in page.get_text("dict")["blocks"]:
                for line in block.get("lines", []):
                    txt = _line_text(line)
                    if txt:
                        out.append((BOLD_MARK.strip() + " " if _line_is_bold(line) else "") + txt)
            pages.append("\n".join(out))
    return "\n".join(strip_page_noise(pages)), first


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def title_on_first_page(title: str, first_page: str) -> bool:
    """Cheap check that the downloaded PDF is the paper we think it is."""
    return _norm(title)[:25] in _norm(first_page)


def ingest(topic: str | None = None, download: bool = True, out: Path = CHUNKS_JSONL, verbose: bool = False) -> list[dict]:
    papers = parse_papers_md()
    if topic:
        papers = [p for p in papers if p["topic"] == topic]
    all_chunks: list[dict] = []
    for p in papers:
        txt, pdf = RAW / f"{p['id']}.txt", RAW / f"{p['id']}.pdf"
        if txt.exists():
            text, first = txt.read_text(encoding="utf-8"), txt.read_text(encoding="utf-8")[:3000]
        else:
            if download and download_pdf(p["arxiv"], pdf):
                time.sleep(3)  # be polite to arXiv
            if not pdf.exists():
                print(f"SKIP {p['id']}: no PDF at {pdf} (download failed or --no-download)")
                continue
            text, first = pdf_to_text(pdf)
        if not title_on_first_page(p["title"], first):
            print(f"WARN {p['id']}: title not found on page 1, check the arXiv id {p['arxiv']}")
        text, garbled = drop_garbled_lines(text)
        if garbled:
            print(f"NOTE {p['id']}: dropped {garbled} garbled lines (broken PDF font encoding)")
        sections = sections_from_text(text)
        chunks = chunk_paper(p, sections)
        all_chunks.extend(chunks)
        names = ", ".join(s for s, _ in sections)
        if verbose:
            print(f"\n{p['id']}: sections={len(sections)} chunks={len(chunks)}")
            for name, body in sections:
                print(f"   {len(body.split()):>5} words  {name}")
        else:
            print(f"{p['id']:<11} sections={len(sections):<3} chunks={len(chunks):<4} [{names[:110]}]")
    write_jsonl(all_chunks, out)
    words = [c["n_words"] for c in all_chunks]
    avg = sum(words) / len(words) if words else 0
    print(
        f"\nDONE papers={len({c['paper_id'] for c in all_chunks})} chunks={len(all_chunks)} "
        f"avg_words={avg:.0f} (chunk={CHUNK_WORDS}, overlap={OVERLAP_WORDS}) -> {out}"
    )
    return all_chunks


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", help="only this topic id")
    ap.add_argument("--no-download", action="store_true", help="use files already in data/raw/")
    ap.add_argument("--verbose", action="store_true", help="print every section with its size")
    args = ap.parse_args()
    ingest(topic=args.topic, download=not args.no_download, verbose=args.verbose)
