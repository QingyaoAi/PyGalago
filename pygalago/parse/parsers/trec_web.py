# BSD License (http://www.galagosearch.org/license)
"""Port of TrecWebParser.java — parse TREC-web (WT10g / GOV2) format.

Format::

    <DOC>
    <DOCNO>WTX001-B01-1</DOCNO>
    <DOCHDR>
    http://www.example.com/ ...
    ...
    </DOCHDR>
    <html>…raw document body…</html>
    </DOC>
"""

from __future__ import annotations

import gzip
from typing import Generator, IO

from pygalago.parse.document import Document


def _open(path: str) -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_file(path: str) -> Generator[Document, None, None]:
    with _open(path) as fh:
        yield from parse_stream(fh)


def parse_stream(stream: IO[str]) -> Generator[Document, None, None]:
    in_doc      = False
    in_dochdr   = False
    docno: str | None = None
    url: str | None = None
    buf: list[str] = []

    for raw_line in stream:
        line    = raw_line.rstrip("\n")
        stripped = line.strip()

        if stripped == "<DOC>":
            in_doc    = True
            in_dochdr = False
            docno     = None
            url       = None
            buf.clear()
            continue

        if stripped == "</DOC>":
            if docno is not None:
                doc = Document(name=docno, text="\n".join(buf))
                if url:
                    doc.metadata["url"] = url
                yield doc
            in_doc = False
            continue

        if not in_doc:
            continue

        if stripped.startswith("<DOCNO>"):
            # <DOCNO>WTX001-B01-1</DOCNO>
            rest = stripped[7:]
            if "</DOCNO>" in rest:
                docno = rest[:rest.index("</DOCNO>")].strip()
            continue

        if stripped == "<DOCHDR>":
            in_dochdr = True
            continue

        if stripped == "</DOCHDR>":
            in_dochdr = False
            continue

        if in_dochdr:
            if url is None:
                # First line in DOCHDR is the URL
                url_line = stripped.split()[0] if stripped else ""
                url = url_line.lower().rstrip("/")
            continue

        # Body content
        buf.append(line)
