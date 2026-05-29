"""Document format parsers."""

import os
from typing import Generator

from pygalago.parse.document import Document


def open_collection(path: str) -> Generator[Document, None, None]:
    """Auto-detect collection format and yield :class:`Document` objects.

    Supported formats:
    - TREC text (``.trec``, ``.txt``, ``.sgml``)
    - TREC web (``.web``)
    - JSON / JSON-Lines (``.json``, ``.jsonl``)
    - WARC (``.warc``)
    - Gzipped variants of all of the above (``.gz``)
    """
    base = path.lower()
    if base.endswith(".gz"):
        base = base[:-3]

    if any(base.endswith(ext) for ext in (".web",)):
        from pygalago.parse.parsers.trec_web import parse_file
        yield from parse_file(path)
    elif any(base.endswith(ext) for ext in (".json", ".jsonl")):
        from pygalago.parse.parsers.json_parser import parse_file
        yield from parse_file(path)
    elif base.endswith(".warc"):
        from pygalago.parse.parsers.warc import parse_file
        yield from parse_file(path)
    else:
        # Default: TREC text format
        from pygalago.parse.parsers.trec_text import parse_file
        yield from parse_file(path)
