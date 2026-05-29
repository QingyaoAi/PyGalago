"""pygalago.retrieval — BM25 ranked retrieval (Phase 3 + Phase 4).

Phase 3 (low-level)::

    results = bm25_search("path/to/index", ["inform", "retriev"], n=1000)

Phase 4 (full pipeline with query parsing and traversals)::

    from pygalago.retrieval import Retrieval

    r = Retrieval("path/to/index")
    results = r.search("information retrieval", n=10)
    for name, score in results:
        print(name, score)

    # Structured query language
    results = r.search("#fdm(information retrieval)", n=10)

    # Explicit #combine with weights
    results = r.search("#combine:0=0.7:1=0.3(information retrieval)", n=10)
"""

from __future__ import annotations

import os
import warnings
from typing import List, Optional, Tuple

from pygalago.query.node            import Node
from pygalago.query.parser          import parse
from pygalago.query.iterator_builder import node_to_weighted_terms
from pygalago.query.traversals      import (
    PartAssignerTraversal,
    AnnotateStatsTraversal,
    FullDependenceTraversal,
)

try:
    from pygalago._galago import (
        bm25_search as _bm25_search_cpp,
        bm25_search_weighted as _bm25_search_weighted_cpp,
        ScoredDocument,
        BM25Params,
        LengthsSource,
        DiskIndex,
        PostingsReader,
    )
    _HAS_EXTENSION = True
except ImportError:
    _HAS_EXTENSION = False
    ScoredDocument = BM25Params = LengthsSource = DiskIndex = PostingsReader = None  # type: ignore


def _require_extension() -> None:
    if not _HAS_EXTENSION:
        raise RuntimeError(
            "The C++ extension (_galago) is not built. "
            "Run `pip install -e .` from the project root."
        )


# ── Low-level Phase-3 API ─────────────────────────────────────────────────────

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

    *terms* must already be normalised/stemmed to match the index part.
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

    Note: no stemming is applied — use the ``Retrieval`` class for a full
    pipeline that handles query parsing and traversals.
    """
    terms = query.lower().split()
    return bm25_search(index_path, terms, b=b, k=k, n=n, part=part)


# ── Full Phase-4 pipeline ─────────────────────────────────────────────────────

class Retrieval:
    """Full retrieval pipeline: parse → traverse → score → rank.

    Mirrors the responsibilities of Galago's ``LocalRetrieval.java``.

    Parameters
    ----------
    index_path:
        Path to a Galago index directory.
    b, k:
        BM25 parameters.
    part:
        Postings index part to use (default: ``"postings.krovetz"``).
    fdm_weights:
        Weights for the Full Dependence Model (SDM) unigram/ordered/unordered
        components.  Keys: ``"uniw"`` (default 0.8), ``"odw"`` (0.15),
        ``"uww"`` (0.05).
    """

    def __init__(
        self,
        index_path: str,
        *,
        b: float = 0.75,
        k: float = 1.2,
        part: str = "postings.krovetz",
        fdm_weights: Optional[dict] = None,
    ) -> None:
        _require_extension()

        self.index_path = index_path
        self.b    = b
        self.k    = k
        self.part = part

        # Open C++ index objects (shared across queries)
        self._index   = DiskIndex(index_path)
        self._lengths = LengthsSource(os.path.join(index_path, "lengths"))

        pr = self._index.postings_reader(part)
        if pr is None:
            raise RuntimeError(
                f"Postings part '{part}' not found in index at {index_path!r}."
            )
        self._pr = pr

        self._ls = self._lengths.stats

        # Traversals
        self._part_assigner = PartAssignerTraversal(part)
        self._annotate      = AnnotateStatsTraversal(
            postings_reader=self._pr,
            lengths_stats=self._ls,
            global_params={"b": b, "k": k},
        )
        self._fdm = FullDependenceTraversal(fdm_weights or {})

    # ── Main search method ────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        n: int = 1000,
        *,
        return_names: bool = True,
    ) -> List[Tuple[str, float]]:
        """Run a full structured query.

        Parameters
        ----------
        query:
            A plain-text or Galago structured query.
        n:
            Maximum number of results to return.
        return_names:
            If True (default) each result is ``(document_name, score)``.
            If False, results are ``(docid: int, score)``.

        Returns
        -------
        list of (name, score) or (docid, score) tuples, best-first.
        """
        weighted_terms = self._process_query(query)
        if not weighted_terms:
            return []

        results = _bm25_search_weighted_cpp(
            self._index, self._lengths, weighted_terms,
            b=self.b, k=self.k, n=n, part=self.part,
        )

        if return_names:
            return [
                (self._index.get_name(r.document), r.score)
                for r in results
            ]
        return [(r.document, r.score) for r in results]

    def search_scored(
        self,
        query: str,
        n: int = 1000,
    ) -> "List[ScoredDocument]":
        """Like :meth:`search` but returns raw :class:`ScoredDocument` objects."""
        weighted_terms = self._process_query(query)
        if not weighted_terms:
            return []
        return _bm25_search_weighted_cpp(
            self._index, self._lengths, weighted_terms,
            b=self.b, k=self.k, n=n, part=self.part,
        )

    # ── Internal pipeline ─────────────────────────────────────────────────────

    def _process_query(self, query: str) -> List[Tuple[str, float]]:
        """Parse and traverse the query, returning weighted term pairs."""
        root = parse(query)

        # Apply traversals in order (matching Galago's default traversal list)
        root = self._part_assigner.traverse(root)
        root = self._fdm.traverse(root)
        root = self._annotate.traverse(root)

        return node_to_weighted_terms(root)

    # ── Diagnostic helpers ────────────────────────────────────────────────────

    def explain(self, query: str) -> dict:
        """Return a debug dict showing the query tree and weighted terms."""
        root_raw  = parse(query)
        root_proc = parse(query)
        root_proc = self._part_assigner.traverse(root_proc)
        root_proc = self._fdm.traverse(root_proc)
        root_proc = self._annotate.traverse(root_proc)
        weighted  = node_to_weighted_terms(root_proc)
        return {
            "raw_tree":       str(root_raw),
            "processed_tree": str(root_proc),
            "weighted_terms": weighted,
        }

    @property
    def total_documents(self) -> int:
        return self._ls.total_document_count

    @property
    def avg_doc_length(self) -> float:
        return self._ls.avg_length


__all__ = [
    "bm25_search",
    "search",
    "Retrieval",
    "ScoredDocument",
    "BM25Params",
    "LengthsSource",
]
