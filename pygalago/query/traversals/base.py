# BSD License (http://www.galagosearch.org/license)
"""Port of Traversal.java — base class for query tree transformations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

from pygalago.query.node import Node


class Traversal(ABC):
    """Abstract base for query tree transformations.

    Subclasses override :meth:`before_node` (pre-order) and/or
    :meth:`after_node` (post-order).  The default implementations are no-ops.

    Call :meth:`traverse` on the root of a query tree to apply the
    transformation recursively.
    """

    def traverse(self, node: Node, params: Dict[str, Any] | None = None) -> Node:
        if params is None:
            params = {}
        node = self.before_node(node, params)
        node.children = [self.traverse(c, params) for c in node.children]
        node = self.after_node(node, params)
        return node

    def before_node(self, node: Node, params: Dict[str, Any]) -> Node:
        return node

    def after_node(self, node: Node, params: Dict[str, Any]) -> Node:
        return node
