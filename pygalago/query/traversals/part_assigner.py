# BSD License (http://www.galagosearch.org/license)
"""Port of PartAssignerTraversal.java.

Assigns the postings index part name (e.g. "postings.krovetz") to every
leaf text node that doesn't already have a "part" parameter.
"""

from __future__ import annotations

from typing import Any, Dict

from pygalago.query.node import Node
from pygalago.query.traversals.base import Traversal

_LEAF_OPS = frozenset(("text", "counts", "extents"))


class PartAssignerTraversal(Traversal):
    """Stamp a 'part' parameter onto every leaf text node."""

    def __init__(self, part: str = "postings.krovetz") -> None:
        self._part = part

    def after_node(self, node: Node, params: Dict[str, Any]) -> Node:
        if node.operator in _LEAF_OPS and "part" not in node.params:
            node.params["part"] = self._part
        return node
