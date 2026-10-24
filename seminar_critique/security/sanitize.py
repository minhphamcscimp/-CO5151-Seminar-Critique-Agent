"""Untrusted content handling. Owner: Tin.

Paper text, retrieved chunks and tool results all come from outside the system.
They may contain instructions aimed at the model (prompt injection).
D2 only asks for the threat model; attack tests belong to the final report.
"""
from __future__ import annotations


def sanitize_untrusted(text: str, source: str) -> str:
    """STUB: wrap external text so the model treats it as data, not instructions.

    TODO (Tin): decide the mitigation per entry point in threat_model.md
    (delimiters, length limit, stripping, instruction hierarchy in the prompt).
    """
    return f"<untrusted source={source!r}>\n{text}\n</untrusted>"
