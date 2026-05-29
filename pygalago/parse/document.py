# BSD License (http://www.galagosearch.org/license)
"""Port of Document.java — the core document representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Tag:
    name: str
    begin: int  # token index where tag opens
    end: int    # token index where tag closes (exclusive)
    attributes: Dict[str, str] = field(default_factory=dict)


@dataclass
class Document:
    """A parsed document ready for indexing.

    Attributes
    ----------
    name:
        External human-readable document identifier (e.g. "FBIS3-1").
    text:
        Raw document text as extracted from the collection format.
    terms:
        Token list after tokenisation and (optionally) stemming.
    tags:
        SGML/HTML tags found in *text*, with token-position ranges.
    metadata:
        Key-value pairs extracted from the collection format
        (e.g. ``{"url": "http://…", "date": "19940104"}``).
    identifier:
        Internal numeric docid assigned by the IndexBuilder.  -1 until built.
    """

    name: str = ""
    text: str = ""
    terms: List[str] = field(default_factory=list)
    tags: List[Tag] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    identifier: int = -1

    @staticmethod
    def from_text(name: str, text: str) -> "Document":
        """Convenience factory matching Java's Document(String, String)."""
        return Document(name=name, text=text)
