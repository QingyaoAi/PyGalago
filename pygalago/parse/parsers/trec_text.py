# BSD License (http://www.galagosearch.org/license)
"""Port of TrecTextParser.java — parse TREC-text format collections.

Format::

    <DOC>
    <DOCNO> FBIS3-1 </DOCNO>
    <TEXT>
    Some document content.
    </TEXT>
    </DOC>

The parser yields :class:`~pygalago.parse.document.Document` objects for
every ``<DOC>…</DOC>`` block, capturing text from recognised content tags.
"""

from __future__ import annotations

import gzip
import io
import os
from typing import Generator, IO, Union

from pygalago.parse.document import Document

_START_TAGS = frozenset((
    "<TEXT>", "<HEADLINE>", "<TITLE>", "<HL>", "<HEAD>",
    "<TTL>", "<DD>", "<DATE>", "<LP>", "<LEADPARA>",
))
_END_TAGS = {
    "<TEXT>":     "</TEXT>",
    "<HEADLINE>": "</HEADLINE>",
    "<TITLE>":    "</TITLE>",
    "<HL>":       "</HL>",
    "<HEAD>":     "</HEAD>",
    "<TTL>":      "</TTL>",
    "<DD>":       "</DD>",
    "<DATE>":     "</DATE>",
    "<LP>":       "</LP>",
    "<LEADPARA>": "</LEADPARA>",
}


def _open(path: str) -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_file(path: str) -> Generator[Document, None, None]:
    """Yield :class:`Document` objects from a TREC-text file (plain or .gz)."""
    with _open(path) as fh:
        yield from parse_stream(fh)


def parse_stream(stream: IO[str]) -> Generator[Document, None, None]:
    """Yield :class:`Document` objects from an open text stream."""
    in_doc     = False
    in_tag: str | None = None
    docno: str | None = None
    buf: list[str] = []

    for raw_line in stream:
        line = raw_line.rstrip("\n")
        stripped = line.strip()

        # ── Document boundary ─────────────────────────────────────────────────
        if stripped == "<DOC>":
            in_doc  = True
            in_tag  = None
            docno   = None
            buf.clear()
            continue

        if stripped == "</DOC>":
            if docno is not None:
                yield Document(name=docno, text="\n".join(buf))
            in_doc  = False
            in_tag  = None
            docno   = None
            buf.clear()
            continue

        if not in_doc:
            continue

        # ── Docno ─────────────────────────────────────────────────────────────
        if stripped.startswith("<DOCNO>"):
            # May be on one line: <DOCNO> FBIS3-1 </DOCNO>
            rest = stripped[7:]
            if "</DOCNO>" in rest:
                docno = rest[:rest.index("</DOCNO>")].strip()
            else:
                # Multi-line — rare but possible
                docno = rest.strip()
            continue

        # ── Content tag tracking ───────────────────────────────────────────────
        if in_tag is not None:
            end_tag = _END_TAGS[in_tag]
            if stripped.startswith(end_tag):
                buf.append(line)
                in_tag = None
            else:
                buf.append(line)
            continue

        # Not inside a content tag
        if stripped.startswith("<"):
            for start in _START_TAGS:
                if stripped.startswith(start):
                    in_tag = start
                    buf.append(line)
                    break
