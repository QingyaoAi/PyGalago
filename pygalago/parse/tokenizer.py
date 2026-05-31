# BSD License (http://www.galagosearch.org/license)
"""Port of TagTokenizer.java (core behaviour) — tokenise document text.

Strips HTML/SGML tags, splits on whitespace and punctuation, lowercases
everything.  This matches Galago's default tokenisation closely enough for
BM25 retrieval without requiring a JVM-linked C port of TagTokenizer.

For maximum compatibility with existing Galago indexes, build indexes with
the same tokenisation settings that were used at index build time.
"""

from __future__ import annotations

import re
from typing import List, Optional

from pygalago.parse.document import Document

# Match contiguous alphanumeric sequences only — matches Java Galago's
# TagTokenizer behaviour of splitting on ALL punctuation including hyphens
# and apostrophes.  Keeping hyphens in word tokens caused prefix strings like
# "non", "anti", "post", "co" to be merged into compounds ("non-proliferation")
# instead of being indexed separately, producing a large vocabulary gap vs
# Galago Java and ~6% MAP loss on Robust04.
_WORD_RE = re.compile(r"[a-zA-Z0-9]+")

# HTML/SGML tag pattern (very permissive).
_TAG_RE = re.compile(r"<[^>]*>")


def tokenize(doc: Document, lowercase: bool = True) -> Document:
    """In-place tokenise *doc.text* → fill *doc.terms*.

    Removes HTML tags, then splits on non-word characters and lowercases.
    Returns *doc* for chaining.
    """
    text = doc.text or ""
    # Strip HTML/SGML tags.
    text = _TAG_RE.sub(" ", text)
    # Extract word-like tokens.
    tokens = _WORD_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    doc.terms = tokens
    return doc


def tokenize_string(text: str, lowercase: bool = True) -> List[str]:
    """Tokenise a bare string and return the token list."""
    text = _TAG_RE.sub(" ", text)
    tokens = _WORD_RE.findall(text)
    if lowercase:
        tokens = [t.lower() for t in tokens]
    return tokens
