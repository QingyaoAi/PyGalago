# BSD License (http://www.galagosearch.org/license)
"""JSON collection parser — one document per line (JSON-Lines) or a JSON array.

Each record must have at least an ``id`` (or ``docno`` / ``docid``) field
and a ``text`` (or ``contents`` / ``body``) field.

Example JSON-Lines format::

    {"id": "doc1", "text": "This is the document body."}
    {"id": "doc2", "text": "Another document."}
"""

from __future__ import annotations

import gzip
import json
from typing import Generator, IO

from pygalago.parse.document import Document

_ID_KEYS   = ("id", "docid", "docno", "identifier")
_TEXT_KEYS = ("text", "contents", "body", "passage")


def _get_field(obj: dict, keys) -> str:
    for k in keys:
        if k in obj:
            return str(obj[k])
    return ""


def _open(path: str) -> IO[str]:
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_file(path: str) -> Generator[Document, None, None]:
    with _open(path) as fh:
        yield from parse_stream(fh)


def parse_stream(stream: IO[str]) -> Generator[Document, None, None]:
    """Parse JSON-Lines or a JSON array from *stream*."""
    raw = stream.read().strip()
    if not raw:
        return

    # Try JSON array first.
    if raw.startswith("["):
        try:
            records = json.loads(raw)
            for r in records:
                doc = _record_to_doc(r)
                if doc:
                    yield doc
            return
        except json.JSONDecodeError:
            pass

    # Fall back to JSON-Lines.
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            doc = _record_to_doc(r)
            if doc:
                yield doc
        except json.JSONDecodeError:
            continue


def _record_to_doc(r: dict) -> Document | None:
    name = _get_field(r, _ID_KEYS)
    text = _get_field(r, _TEXT_KEYS)
    if not name:
        return None
    metadata = {k: str(v) for k, v in r.items()
                if k not in _ID_KEYS and k not in _TEXT_KEYS}
    return Document(name=name, text=text, metadata=metadata)
