# BSD License (http://www.galagosearch.org/license)
"""Port of Node.java — the query tree node.

A Node represents a single operator or leaf term in a Galago structured query.
Examples of query strings and their tree representations:

    "term"
        Node(operator='text', params={'default': 'term'}, children=[])

    "#combine(t1 t2)"
        Node('combine', {}, [Node('text', {'default': 't1'}),
                              Node('text', {'default': 't2'})])

    "#weight:0=0.8:1=0.2(t1 t2)"
        Node('combine', {'0': 0.8, '1': 0.2}, [Node('text', {'default': 't1'}),
                                                 Node('text', {'default': 't2'})])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Node:
    """A node in a Galago structured query tree."""

    operator: str
    params: Dict[str, Any] = field(default_factory=dict)
    children: List["Node"] = field(default_factory=list)
    position: int = 0

    # ── Factory methods ───────────────────────────────────────────────────────

    @staticmethod
    def text(term: str, pos: int = 0) -> "Node":
        """Create a leaf text node for a single query term."""
        return Node(operator="text", params={"default": term}, position=pos)

    # ── Convenience accessors ─────────────────────────────────────────────────

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @property
    def default_parameter(self) -> str:
        """The positional (un-named) node parameter, e.g. the term string."""
        return str(self.params.get("default", ""))

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    # ── Serialisation ─────────────────────────────────────────────────────────

    def __str__(self) -> str:
        return self._to_string()

    def _to_string(self) -> str:
        if self.operator == "text":
            term = self.default_parameter
            if any(c in term for c in " ()#:"):
                return f'"{term}"'
            return term

        # Build parameter string:  :key=value  or  :value (for 'default')
        param_parts: List[str] = []
        for k, v in sorted(self.params.items()):
            if k == "default":
                param_parts.append(f":{_fmt(v)}")
            else:
                param_parts.append(f":{k}={_fmt(v)}")

        params_str = "".join(param_parts)
        children_str = " ".join(c._to_string() for c in self.children)
        return f"#{self.operator}{params_str}({children_str})"

    def to_pretty_string(self, indent: int = 0) -> str:
        pad = "  " * indent
        if self.operator == "text":
            return f"{pad}{self.default_parameter}"
        lines = [f"{pad}#{self.operator}"]
        for c in self.children:
            lines.append(c.to_pretty_string(indent + 1))
        return "\n".join(lines)

    # ── Cloning ───────────────────────────────────────────────────────────────

    def clone(self) -> "Node":
        return Node(
            operator=self.operator,
            params=dict(self.params),
            children=[c.clone() for c in self.children],
            position=self.position,
        )


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        # Match Java's DecimalFormat behaviour: up to 6 sig figs, no trailing zeros
        s = f"{v:.6g}"
        return s
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
