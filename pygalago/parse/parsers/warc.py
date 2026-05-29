# BSD License (http://www.galagosearch.org/license)
"""Port of WARCParser.java — parse WARC (Web ARChive) format collections.

Supports WARC/1.0 and WARC/1.1 files (plain or gzip-compressed).
Only ``WARC-Type: response`` records that contain an HTTP body are yielded.

The document name is taken from ``WARC-TREC-ID`` if present, falling back to
``WARC-Target-URI``.  The document text is the HTTP response body (everything
after the blank line following the HTTP status line).
"""

from __future__ import annotations

import gzip
import re
from typing import Generator, IO

from pygalago.parse.document import Document

_WARC_VERSION_RE = re.compile(r"^WARC/[0-9.]+\s*$", re.IGNORECASE)
_CONTENT_LEN_RE  = re.compile(r"^Content-Length:\s*(\d+)", re.IGNORECASE)
_WARC_TYPE_RE    = re.compile(r"^WARC-Type:\s*(\S+)", re.IGNORECASE)
_WARC_URI_RE     = re.compile(r"^WARC-Target-URI:\s*(\S+)", re.IGNORECASE)
_WARC_TRECID_RE  = re.compile(r"^WARC-TREC-ID:\s*(\S+)", re.IGNORECASE)


def _open(path: str) -> IO[bytes]:
    if path.endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def parse_file(path: str) -> Generator[Document, None, None]:
    """Yield :class:`Document` objects from a WARC file (plain or .gz)."""
    with _open(path) as fh:
        yield from parse_stream(fh)


def parse_stream(stream: IO[bytes]) -> Generator[Document, None, None]:
    """Yield :class:`Document` objects from an open binary WARC stream."""
    while True:
        record = _read_record(stream)
        if record is None:
            break
        doc = _record_to_doc(record)
        if doc is not None:
            yield doc


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read_line(stream: IO[bytes]) -> bytes | None:
    line = stream.readline()
    return line if line else None


def _read_record(stream: IO[bytes]) -> dict | None:
    """Read one WARC record and return a dict, or None at EOF."""
    # Skip blank lines between records
    while True:
        line = _read_line(stream)
        if line is None:
            return None
        stripped = line.rstrip(b"\r\n")
        if stripped and _WARC_VERSION_RE.match(stripped.decode("ascii", "replace")):
            break

    # Parse WARC headers until blank line
    headers: dict[str, str] = {}
    content_length = 0
    while True:
        line = _read_line(stream)
        if line is None:
            return None
        stripped = line.rstrip(b"\r\n")
        if not stripped:
            break
        text = stripped.decode("utf-8", "replace")
        m = _CONTENT_LEN_RE.match(text)
        if m:
            content_length = int(m.group(1))
        m = _WARC_TYPE_RE.match(text)
        if m:
            headers["type"] = m.group(1).lower()
        m = _WARC_URI_RE.match(text)
        if m:
            headers["uri"] = m.group(1)
        m = _WARC_TRECID_RE.match(text)
        if m:
            headers["trec_id"] = m.group(1)

    # Read exactly content_length bytes
    body = b""
    if content_length > 0:
        body = stream.read(content_length)

    # Consume the trailing \r\n\r\n after record body
    stream.read(4)

    return {"headers": headers, "body": body}


def _record_to_doc(record: dict) -> Document | None:
    """Convert a parsed WARC record dict to a Document, or None if skipping."""
    headers = record["headers"]
    if headers.get("type") != "response":
        return None

    body_bytes: bytes = record["body"]

    # Strip HTTP response headers — find the blank line separator
    try:
        sep = body_bytes.index(b"\r\n\r\n")
        http_body = body_bytes[sep + 4:]
    except ValueError:
        try:
            sep = body_bytes.index(b"\n\n")
            http_body = body_bytes[sep + 2:]
        except ValueError:
            http_body = body_bytes

    text = http_body.decode("utf-8", "replace")

    name = headers.get("trec_id") or headers.get("uri") or ""
    if not name:
        return None

    doc = Document(name=name, text=text)
    if "uri" in headers:
        doc.metadata["url"] = headers["uri"]
    return doc
