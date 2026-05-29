"""pygalago.index — index reading (Phase 2).

Public API::

    import pygalago.index as gi

    # Open a full index directory
    idx = gi.open("/path/to/index")
    print(idx.get_name(0))           # "doc0001"
    print(idx.get_length(0))         # 312
    stats = idx.get_length_stats()   # LengthStats object
    for docid, count in idx.get_postings("information"):
        print(docid, count)

    # Or open individual parts directly
    names   = gi.open_names("/path/to/index/names")
    lengths = gi.open_lengths("/path/to/index/lengths")
    posts   = gi.open_postings("/path/to/index/postings.krovetz")
"""

from __future__ import annotations

try:
    from pygalago._galago import (
        DiskIndex,
        NamesReader,
        LengthsReader,
        LengthStats,
        PostingsReader,
        PostingsIterator,
        BTreeReader,
    )
    _HAS_EXTENSION = True
except ImportError:
    _HAS_EXTENSION = False
    DiskIndex = NamesReader = LengthsReader = LengthStats = None  # type: ignore
    PostingsReader = PostingsIterator = BTreeReader = None  # type: ignore


def _require_extension() -> None:
    if not _HAS_EXTENSION:
        raise RuntimeError(
            "The C++ extension (_galago) is not built. "
            "Run `pip install -e .` from the project root."
        )


def open(path: str) -> "DiskIndex":
    """Open a Galago index directory and return a DiskIndex."""
    _require_extension()
    return DiskIndex(path)


def open_names(path: str) -> "NamesReader":
    """Open a Galago `names` B-tree file directly."""
    _require_extension()
    return NamesReader(path)


def open_lengths(path: str) -> "LengthsReader":
    """Open a Galago `lengths` B-tree file directly."""
    _require_extension()
    return LengthsReader(path)


def open_postings(path: str) -> "PostingsReader":
    """Open a Galago postings B-tree file directly."""
    _require_extension()
    return PostingsReader(path)


def open_btree(path: str) -> "BTreeReader":
    """Open a raw Galago B-tree file for low-level iteration."""
    _require_extension()
    return BTreeReader(path)


__all__ = [
    "open",
    "open_names",
    "open_lengths",
    "open_postings",
    "open_btree",
    "DiskIndex",
    "NamesReader",
    "LengthsReader",
    "LengthStats",
    "PostingsReader",
    "PostingsIterator",
    "BTreeReader",
]
