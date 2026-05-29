"""pygalago.query — query parsing and tree representation.

Quick start::

    from pygalago.query import parse, Node

    node = parse("information retrieval")
    print(node)   # #combine(information retrieval)

    node = parse("#fdm(information retrieval)")
    print(node)   # #fdm(information retrieval)
"""

from pygalago.query.node   import Node
from pygalago.query.parser import parse, find_query_terms

__all__ = ["Node", "parse", "find_query_terms"]
