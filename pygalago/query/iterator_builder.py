# BSD License (http://www.galagosearch.org/license)
"""Convert an annotated Node tree into C++ iterator input.

After traversals have annotated the Node tree with index statistics,
this module flattens the tree into a list of (term, effective_weight)
pairs that ``bm25_search_weighted`` can consume.

Supported operators
-------------------
text / counts
    Leaf: contributes one (term, weight) pair.
combine
    OR-combine children with optional per-index weights (params["0"], …).
    Weights default to uniform 1/N and are *multiplied* into the child weight.
weight
    Synonym for ``combine`` with explicit numeric weights.
ordered / od / unordered / uw
    Proximity operators — require the positional index (Phase 5).
    Currently emitted as ``None`` (skipped) with a logged warning.

The returned list is de-duplicated: if a term appears multiple times (e.g.
in both the unigram and bigram components of an SDM expansion), its weights
are summed.  The final weights are then re-normalised so they sum to 1.
"""

from __future__ import annotations

import logging
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pygalago.query.node import Node

logger = logging.getLogger(__name__)

# Operators whose children can be recursed into.
_COMBINE_OPS  = frozenset(("combine", "weight", "root"))
# Leaf operators that produce a BM25 term.
_LEAF_OPS     = frozenset(("text", "counts", "extents"))
# Proximity operators — not yet supported; skipped with a warning.
_PROXY_OPS    = frozenset(("ordered", "od", "unordered", "uw",
                             "extentinside", "inside", "smoothinside"))


def node_to_weighted_terms(
    node: Node,
    *,
    warn_unsupported: bool = True,
) -> List[Tuple[str, float]]:
    """Flatten *node* tree to ``[(term, weight), …]`` pairs (normalised).

    Proximity nodes (``#ordered``, ``#unordered``) are silently dropped and
    the remaining weights are re-normalised.
    """
    raw: Dict[str, float] = defaultdict(float)
    _collect(node, weight=1.0, out=raw, warn=warn_unsupported)

    if not raw:
        return []

    # Normalise
    total = sum(raw.values())
    if total <= 0.0:
        return []
    return [(t, w / total) for t, w in sorted(raw.items())]


def _collect(node: Node, weight: float, out: Dict[str, float],
             warn: bool) -> None:
    """Recursive helper; accumulates (term → effective_weight) into *out*."""

    op = node.operator

    if op in _LEAF_OPS:
        term = node.default_parameter
        if term:
            out[term] += weight
        return

    if op in _COMBINE_OPS:
        n = len(node.children)
        if n == 0:
            return
        # Read explicit per-child weights from node parameters.
        child_weights: List[float] = []
        for i in range(n):
            w = node.params.get(str(i), None)
            child_weights.append(float(w) if w is not None else 1.0)
        total_w = sum(child_weights)
        if total_w <= 0.0:
            total_w = 1.0
        for i, child in enumerate(node.children):
            eff_w = weight * (child_weights[i] / total_w)
            _collect(child, eff_w, out, warn)
        return

    if op in _PROXY_OPS:
        if warn:
            warnings.warn(
                f"Proximity operator '#{op}' requires the positional index "
                f"(not yet implemented — skipping this component). "
                f"Results use BM25 bag-of-words only.",
                UserWarning,
                stacklevel=4,
            )
        return

    # Unrecognised operator — recurse into children with full weight split.
    n = len(node.children)
    if n == 0:
        return
    sub_w = weight / n
    for child in node.children:
        _collect(child, sub_w, out, warn)
