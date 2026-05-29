# BSD License (http://www.galagosearch.org/license)
"""Port of AnnotateParameters.java — attaches BM25 statistics to leaf nodes.

For each leaf text/counts node the traversal reads from the index:
  collectionLength   — total token count across all documents
  documentCount      — total number of documents
  nodeDocumentCount  — df (documents containing this term)
  nodeFrequency      — cf (total occurrences of this term)
  maximumCount       — max term frequency in any single document
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pygalago.query.node import Node
from pygalago.query.traversals.base import Traversal

_LEAF_OPS = frozenset(("text", "counts", "extents"))

# Keys that must be present on a leaf node for BM25 scoring.
_BM25_STATS = ("collectionLength", "documentCount",
               "nodeDocumentCount", "nodeFrequency", "maximumCount")


class AnnotateStatsTraversal(Traversal):
    """Attach index statistics to every leaf text/counts node.

    Parameters
    ----------
    postings_reader:
        A ``pygalago._galago.PostingsReader`` (C++ object).  Used to look up
        per-term statistics (df, cf, max_tf).
    lengths_stats:
        A ``pygalago._galago.LengthStats`` object supplying collection-level
        statistics (N, collection_length, avg_dl).
    global_params:
        Optional dict of default BM25 hyper-parameters (``b``, ``k``, …)
        that get copied onto nodes which don't already have them set.
    """

    def __init__(self, postings_reader, lengths_stats, global_params=None):
        self._pr     = postings_reader
        self._ls     = lengths_stats
        self._global = global_params or {}

    def after_node(self, node: Node, params: Dict[str, Any]) -> Node:
        if node.operator not in _LEAF_OPS:
            return node

        term = node.default_parameter
        if not term:
            return node

        # Annotate collection-level stats (always available).
        if "collectionLength" not in node.params:
            node.params["collectionLength"] = self._ls.collection_length
        if "documentCount" not in node.params:
            node.params["documentCount"] = self._ls.total_document_count

        # Per-term stats — look up in the index.
        term_stats = self._pr.get_stats(term)
        if term_stats is not None:
            if "nodeDocumentCount" not in node.params:
                node.params["nodeDocumentCount"] = term_stats["document_count"]
            if "nodeFrequency" not in node.params:
                node.params["nodeFrequency"] = term_stats["collection_count"]
            if "maximumCount" not in node.params:
                node.params["maximumCount"] = term_stats.get("max_tf", 0)

        # Copy global defaults (b, k, etc.) onto node if not already set.
        for key, val in self._global.items():
            if key not in node.params:
                node.params[key] = val

        return node
