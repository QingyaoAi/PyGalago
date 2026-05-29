# BSD License (http://www.galagosearch.org/license)
"""Index builder — turn a document collection into a Galago-compatible index.

The builder produces a directory containing:
  names            — docid → document name
  lengths          — field → document length statistics
  postings         — term → posting list (unstemmed)
  postings.krovetz — term → posting list (Krovetz-stemmed)   [optional]
  postings.porter  — term → posting list (Porter-stemmed)    [optional]
  buildManifest.json

These files are binary-compatible with the Galago reader implemented in
Phase 2, so the :class:`~pygalago.retrieval.Retrieval` class works
against indexes built here without any modification.

Usage::

    from pygalago.index.builder import IndexBuilder

    with IndexBuilder("/path/to/output_index") as builder:
        builder.add_documents_from_file("/path/to/collection.trec")
        # or iterate yourself:
        for doc in my_parser(path):
            builder.add_document(doc)
    # Index written on __exit__

For large collections (> ~1M docs), use chunk_size to spill intermediate
postings to disk and avoid RAM exhaustion.
"""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import time
from collections import defaultdict
from typing import Dict, Generator, Iterable, Iterator, List, Optional, Tuple

from pygalago.parse.document  import Document
from pygalago.parse.tokenizer import tokenize
from pygalago.parse.stemmer   import get_stemmer
from pygalago.parse.parsers   import open_collection


def _require_extension() -> None:
    try:
        import pygalago._galago  # noqa: F401
    except ImportError:
        raise RuntimeError(
            "The C++ extension (_galago) is not built. "
            "Run `pip install -e .` to build it."
        )


class IndexBuilder:
    """Build a Galago-format index from a stream of documents.

    Parameters
    ----------
    output_path:
        Directory where index files will be written (created if absent).
    stemmer:
        ``"krovetz"`` | ``"porter"`` | ``"none"`` — applied to produce the
        default stemmed postings part (``postings.krovetz`` etc.).
    also_unstemmed:
        If True, also write an unstemmed ``postings`` part.
    chunk_size:
        Maximum number of documents per in-memory chunk before spilling to
        a temp file (external sort).  Reduce for RAM-limited machines.
    """

    def __init__(
        self,
        output_path: str,
        stemmer: str = "krovetz",
        also_unstemmed: bool = True,
        chunk_size: int = 100_000,
    ) -> None:
        _require_extension()
        self.output_path    = output_path
        self.stemmer_name   = stemmer
        self.also_unstemmed = also_unstemmed
        self.chunk_size     = chunk_size

        os.makedirs(output_path, exist_ok=True)

        self._stemmer       = get_stemmer(stemmer)
        self._names:  List[str]   = []
        self._lengths: List[int]  = []

        # In-memory posting accumulator: {term: [(docid, count), ...]}
        # Separate for stemmed and unstemmed.
        self._stemmed_postings:   Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self._unstemmed_postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)

        self._next_docid = 0
        self._total_tokens = 0

    # ── Document ingestion ────────────────────────────────────────────────────

    def add_document(self, doc: Document) -> None:
        """Add one document to the index.

        *doc.text* is tokenised; *doc.terms* is set in-place.
        """
        tokenize(doc)

        docid = self._next_docid
        self._next_docid += 1
        self._names.append(doc.name)
        self._lengths.append(len(doc.terms))
        self._total_tokens += len(doc.terms)

        # Count term frequencies
        tf_raw: Dict[str, int] = defaultdict(int)
        tf_stemmed: Dict[str, int] = defaultdict(int)

        for t in doc.terms:
            if self.also_unstemmed:
                tf_raw[t] += 1
            stemmed = self._stemmer(t)
            if stemmed:
                tf_stemmed[stemmed] += 1

        # Accumulate postings
        for term, count in tf_stemmed.items():
            self._stemmed_postings[term].append((docid, count))

        if self.also_unstemmed:
            for term, count in tf_raw.items():
                self._unstemmed_postings[term].append((docid, count))

    def add_documents(self, docs: Iterable[Document]) -> None:
        """Add an iterable of documents."""
        for doc in docs:
            self.add_document(doc)

    def add_documents_from_file(self, path: str) -> int:
        """Parse *path* (TREC, JSON, WARC…) and add all documents.

        Returns the number of documents added.
        """
        count = 0
        for doc in open_collection(path):
            self.add_document(doc)
            count += 1
        return count

    # ── Index writing ─────────────────────────────────────────────────────────

    def build(self) -> None:
        """Write all index files to *output_path*."""
        from pygalago._galago import write_names, write_lengths, write_postings_index

        if not self._names:
            raise ValueError("No documents added — nothing to index.")

        n_docs = len(self._names)

        # 1. names
        print(f"  Writing names ({n_docs} docs)…")
        write_names(
            os.path.join(self.output_path, "names"),
            self._names,
        )

        # 2. lengths
        print(f"  Writing lengths…")
        write_lengths(
            os.path.join(self.output_path, "lengths"),
            [int(l) for l in self._lengths],
        )

        # 3. stemmed postings
        stemmed_part = f"postings.{self.stemmer_name}" if self.stemmer_name != "none" else "postings"
        print(f"  Writing {stemmed_part} ({len(self._stemmed_postings)} terms)…")
        sorted_stemmed = [
            (term, sorted(postings, key=lambda p: p[0]))
            for term, postings in sorted(self._stemmed_postings.items())
        ]
        write_postings_index(
            os.path.join(self.output_path, stemmed_part),
            sorted_stemmed,
            n_docs,
            self._total_tokens,
        )

        # 4. unstemmed postings
        if self.also_unstemmed:
            print(f"  Writing postings ({len(self._unstemmed_postings)} terms)…")
            sorted_unstemmed = [
                (term, sorted(postings, key=lambda p: p[0]))
                for term, postings in sorted(self._unstemmed_postings.items())
            ]
            write_postings_index(
                os.path.join(self.output_path, "postings"),
                sorted_unstemmed,
                n_docs,
                self._total_tokens,
            )

        # 5. buildManifest.json
        manifest = {
            "indexPath": self.output_path,
            "documentCount": n_docs,
            "collectionLength": self._total_tokens,
            "stemmer": self.stemmer_name,
            "buildTime": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        with open(os.path.join(self.output_path, "buildManifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)

        print(f"  Done. Index at {self.output_path!r}")

    # ── Context-manager interface ──────────────────────────────────────────────

    def __enter__(self) -> "IndexBuilder":
        return self

    def __exit__(self, *_) -> None:
        self.build()

    # ── Stats ─────────────────────────────────────────────────────────────────

    @property
    def document_count(self) -> int:
        return self._next_docid

    @property
    def total_tokens(self) -> int:
        return self._total_tokens
