"""pygalago.retrieval — BM25 ranked retrieval (Phase 3).

Quick start::

    import pygalago.retrieval as gr

    # Multi-term BM25 search (terms must already be Krovetz-stemmed)
    results = gr.bm25_search("path/to/index", ["inform", "retriev"], n=1000)
    for r in results[:5]:
        print(r.document, r.score)

    # Simple wrapper that handles tokenisation (lowercase + Krovetz stub)
    results = gr.search("path/to/index", "information retrieval", n=1000)
"""

from __future__ import annotations

from typing import List, Optional

try:
    from pygalago._galago import (
        bm25_search as _bm25_search_cpp,
        ScoredDocument,
        BM25Params,
        LengthsSource,
    )
    _HAS_EXTENSION = True
except ImportError:
    _HAS_EXTENSION = False
    ScoredDocument = BM25Params = LengthsSource = None  # type: ignore


def _require_extension() -> None:
    if not _HAS_EXTENSION:
        raise RuntimeError(
            "The C++ extension (_galago) is not built. "
            "Run `pip install -e .` from the project root."
        )


def bm25_search(
    index_path: str,
    terms: List[str],
    *,
    b: float = 0.75,
    k: float = 1.2,
    n: int = 1000,
    part: str = "postings.krovetz",
) -> "List[ScoredDocument]":
    """Run DAAT BM25 retrieval over *index_path*.

    *terms* must already be normalised/stemmed to match whatever was applied
    at index-build time (e.g. Krovetz stems for ``postings.krovetz``).

    Returns a list of :class:`ScoredDocument` sorted by descending score.
    """
    _require_extension()
    return _bm25_search_cpp(index_path, terms, b=b, k=k, n=n, part=part)


def search(
    index_path: str,
    query: str,
    *,
    b: float = 0.75,
    k: float = 1.2,
    n: int = 1000,
    part: str = "postings.krovetz",
) -> "List[ScoredDocument]":
    """Tokenise *query* (whitespace split + lowercase) and run BM25.

    Note: this does *not* apply Krovetz stemming — use the raw
    ``postings`` part if you want unstemmed matching, or pass
    pre-stemmed tokens to :func:`bm25_search` directly.
    """
    terms = query.lower().split()
    return bm25_search(index_path, terms, b=b, k=k, n=n, part=part)


__all__ = [
    "bm25_search",
    "search",
    "ScoredDocument",
    "BM25Params",
    "LengthsSource",
]
