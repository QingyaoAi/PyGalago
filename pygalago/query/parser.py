# BSD License (http://www.galagosearch.org/license)
"""Port of StructuredLexer.java + StructuredQuery.java.

Parses Galago's structured query language into a Node tree.

Grammar::

    query    ::= argument+
    argument ::= restricted
    restricted ::= unrestricted ('.' field_list)*
    unrestricted ::= operator | term
    operator ::= '#' name params? '(' argument* ')'
    params   ::= (':' key ('=' value)?)+
    term     ::= '"' chars '"' | chars
    key,value ::= printable chars (no space, colon, equals, paren)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Optional

from pygalago.query.node import Node


# ── Lexer ─────────────────────────────────────────────────────────────────────

class TT(Enum):
    TERM         = auto()
    QUOTE        = auto()   # double-quoted string
    SPECIALQUOTE = auto()   # @/.../ string


@dataclass
class Token:
    text: str
    position: int
    ttype: TT = TT.TERM


_SPECIALS = set("#:=()\",.")


def _tokenize(query: str) -> List[Token]:
    """Port of StructuredLexer.tokens()."""
    tokens: List[Token] = []
    start = 0
    i = 0
    n = len(query)

    while i < n:
        c = query[i]
        is_special = c in _SPECIALS or c == "@"
        is_space   = c.isspace()

        if is_special or is_space:
            if start != i:
                tokens.append(Token(query[start:i], start, TT.TERM))

            if c == "@":
                if i + 1 < n:
                    esc_char = query[i + 1]
                    end = query.find(esc_char, i + 2)
                    if end < 0:
                        raise SyntaxError(f"Unmatched '@' escape at position {i}")
                    tokens.append(Token(query[i + 2:end], i, TT.SPECIALQUOTE))
                    i = end
                else:
                    raise SyntaxError(f"'@' at end of query")

            elif c == '"':
                end = query.find('"', i + 1)
                if end < 0:
                    raise SyntaxError(f"Unclosed '\"' at position {i}")
                quoted = query[i + 1:end]
                tokens.append(Token('"',    i,   TT.QUOTE))
                tokens.append(Token(quoted, i+1, TT.QUOTE))
                tokens.append(Token('"',    end, TT.QUOTE))
                i = end

            elif not is_space:
                # Skip a trailing dot that is not followed by a field spec
                if c == ".":
                    bad_dot = (i + 1 >= n) or query[i + 1].isspace()
                    if bad_dot:
                        start = i + 1
                        i += 1
                        continue
                tokens.append(Token(c, i, TT.TERM))

            start = i + 1
        i += 1

    if start < n:
        tokens.append(Token(query[start:], start, TT.TERM))

    return tokens


class _Stream:
    """Token stream with a mark/rewind stack (port of TokenStream)."""

    def __init__(self, tokens: List[Token]) -> None:
        self._tokens = tokens
        self._idx = 0
        self._marks: List[int] = []

    def has_current(self) -> bool:
        return self._idx < len(self._tokens)

    def current(self) -> Optional[Token]:
        return self._tokens[self._idx] if self.has_current() else None

    def current_equals(self, s: str) -> bool:
        t = self.current()
        return t is not None and t.text == s

    def next(self) -> bool:
        self._idx += 1
        return self.has_current()

    def push_mark(self) -> None:
        self._marks.append(self._idx)

    def pop_mark(self) -> None:
        self._marks.pop()

    def rewind_to_mark(self) -> None:
        self._idx = self._marks.pop()

    def reset_mark(self) -> None:
        self._marks.pop()
        self._marks.append(self._idx)


# ── Parser (port of StructuredQuery) ─────────────────────────────────────────

def _parse_parameter_term(stream: _Stream) -> Token:
    t = stream.current()
    tok = Token(t.text, t.position, t.ttype)
    stream.next()
    while (stream.has_current()
           and not stream.current_equals(":")
           and not stream.current_equals("=")
           and not stream.current_equals("(")):
        cur = stream.current()
        tok.text += cur.text
        if cur.ttype != TT.TERM:
            tok.ttype = cur.ttype
        stream.next()
    return tok


def _parse_params(stream: _Stream) -> dict:
    """Parse :key=value or :value pairs into a dict."""
    params: dict = {}
    while stream.current_equals(":"):
        stream.next()
        key_tok = _parse_parameter_term(stream)
        if stream.current_equals("="):
            stream.next()
            val_tok = _parse_parameter_term(stream)
        else:
            val_tok = key_tok
            key_tok = Token("default", key_tok.position, TT.TERM)

        key = key_tok.text
        raw = val_tok.text
        if val_tok.ttype != TT.TERM:
            params[key] = raw          # quoted → always string
        else:
            params[key] = _coerce(raw) # infer bool/int/float/str
    return params


def _coerce(s: str):
    """Type-coerce a NodeParameters value string."""
    if s.lower() == "true":  return True
    if s.lower() == "false": return False
    try:
        i = int(s);  return i
    except ValueError:
        pass
    try:
        f = float(s); return f
    except ValueError:
        pass
    return s


def _parse_operator(stream: _Stream) -> Node:
    pos = stream.current().position
    assert stream.current_equals("#")
    stream.next()

    op_name = stream.current().text
    stream.next()

    params: dict = {}
    if stream.current_equals(":"):
        params = _parse_params(stream)

    if stream.current_equals("("):
        stream.next()

    children = _parse_argument_list(stream)

    if stream.current_equals(")"):
        stream.next()

    return Node(operator=op_name, params=params, children=children, position=pos)


def _parse_quoted_term(stream: _Stream) -> Node:
    assert stream.current_equals('"')
    pos = stream.current().position
    stream.next()
    text = stream.current().text
    stream.next()
    if stream.current_equals('"'):
        stream.next()
    return Node.text(text, pos)


def _parse_term(stream: _Stream) -> Node:
    if stream.current_equals('"'):
        return _parse_quoted_term(stream)
    t = stream.current()
    node = Node.text(t.text, t.position)
    stream.next()
    return node


def _node_with_optional_extent_or(op: str, child: Node, fields: List[Node]) -> Node:
    second = fields[0] if len(fields) == 1 else Node("extentor", {}, fields)
    return Node(op, {}, [child, second])


def _parse_field_list(stream: _Stream) -> List[Node]:
    nodes: List[Node] = []
    t = stream.current()
    nodes.append(Node("field", {"default": t.text}, [], t.position))
    stream.next()
    while stream.current_equals(","):
        stream.next()
        t = stream.current()
        nodes.append(Node("field", {"default": t.text}, [], t.position))
        stream.next()
    return nodes


def _parse_unrestricted(stream: _Stream) -> Node:
    if stream.current_equals("#"):
        return _parse_operator(stream)
    return _parse_term(stream)


def _parse_restricted(stream: _Stream) -> Node:
    node = _parse_unrestricted(stream)
    stream.push_mark()
    while stream.has_current() and stream.current_equals("."):
        stream.next()
        if stream.current_equals("("):
            break  # not a field restriction
        fields = _parse_field_list(stream)
        node = _node_with_optional_extent_or("inside", node, fields)
        stream.reset_mark()
    stream.rewind_to_mark()
    return node


def _parse_argument(stream: _Stream) -> Node:
    node = _parse_restricted(stream)
    if stream.current_equals("."):
        stream.next()
        assert stream.current_equals("(")
        stream.next()
        fields = _parse_field_list(stream)
        assert stream.current_equals(")")
        stream.next()
        node = _node_with_optional_extent_or("smoothinside", node, fields)
    return node


def _parse_argument_list(stream: _Stream) -> List[Node]:
    args: List[Node] = []
    while stream.has_current() and not stream.current_equals(")"):
        args.append(_parse_argument(stream))
    return args


# ── Public API ────────────────────────────────────────────────────────────────

def parse(query: str) -> Node:
    """Parse a Galago structured query string into a Node tree.

    A bare multi-word query like ``"information retrieval"`` is wrapped in a
    ``#combine`` node so the tree always has a single root.
    """
    tokens = _tokenize(query.strip())
    if not tokens:
        return Node.text("")

    stream = _Stream(tokens)
    args = _parse_argument_list(stream)

    if not args:
        return Node.text("")
    if len(args) == 1:
        return args[0]
    # Multiple top-level terms → implicit #combine
    return Node("combine", {}, args, 0)


def find_query_terms(node: Node) -> set:
    """Recursively collect all leaf term strings from a Node tree."""
    if node.operator in ("text", "counts", "extents"):
        term = node.default_parameter
        return {term} if term else set()
    result: set = set()
    for c in node.children:
        result |= find_query_terms(c)
    return result
