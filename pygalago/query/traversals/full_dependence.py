# BSD License (http://www.galagosearch.org/license)
"""Port of FullDependenceTraversal.java — Sequential Dependence Model (SDM).

Expands a ``#fdm`` / ``#fulldep`` operator into:

    #combine:0=<uni_w>:1=<od_w>:2=<uw_w>(
        #combine(<unigrams>)
        #combine(#od:1(t1 t2) #od:1(t1 t3) … )
        #combine(#uw:<4n>(t1 t2) … )
    )

Default weights: unigram=0.8, ordered=0.15, unordered=0.05.

The ordered and unordered window components require the positional index
(``postings.krovetz`` with position data) which is not yet implemented.
When those components are present in the tree, the retrieval engine skips
them with a warning and re-normalises the remaining scores.
"""

from __future__ import annotations

from itertools import combinations
from typing import Any, Dict, List

from pygalago.query.node import Node
from pygalago.query.traversals.base import Traversal


class FullDependenceTraversal(Traversal):
    """Expand #fdm / #fulldep → weighted unigram + ordered + unordered nodes."""

    def __init__(self, params: Dict[str, Any] | None = None) -> None:
        p = params or {}
        self._uni_w    = float(p.get("uniw",        0.8))
        self._od_w     = float(p.get("odw",         0.15))
        self._uw_w     = float(p.get("uww",         0.05))
        self._win_lim  = int(  p.get("windowLimit", 3))

    def after_node(self, node: Node, params: Dict[str, Any]) -> Node:
        if node.operator not in ("fdm", "fulldep"):
            return node

        # Read per-node overrides.
        uni_w   = float(node.params.get("uniw",        self._uni_w))
        od_w    = float(node.params.get("odw",         self._od_w))
        uw_w    = float(node.params.get("uww",         self._uw_w))
        win_lim = int(  node.params.get("windowLimit", self._win_lim))

        children = node.children
        if not children:
            return Node.text("")

        # Unigram component: #combine(t1 t2 … tk)
        unigrams = Node("combine", {}, [c.clone() for c in children])
        if len(children) == 1:
            return unigrams

        # Ordered and unordered window components from power set of terms.
        ordered_nodes:   List[Node] = []
        unordered_nodes: List[Node] = []

        for r in range(2, len(children) + 1):
            if win_lim >= 2 and r > win_lim:
                break
            for combo in combinations(children, r):
                uw_size = 4 * len(combo)
                cloned  = [c.clone() for c in combo]
                ordered_nodes.append(
                    Node("ordered",   {"default": 1},       cloned))
                unordered_nodes.append(
                    Node("unordered", {"default": uw_size}, [c.clone() for c in combo]))

        od_node = Node("combine", {}, ordered_nodes)
        uw_node = Node("combine", {}, unordered_nodes)

        weights = {"0": uni_w, "1": od_w, "2": uw_w}
        return Node("combine", weights, [unigrams, od_node, uw_node],
                    node.position)
