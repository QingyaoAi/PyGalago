"""pygalago.index — index reading and building.

Currently exposes the C++ DiskBTreeReader via the _galago extension if built,
and will grow to include the full index reading layer (Phase 2).
"""

from __future__ import annotations

from typing import Optional

try:
    from pygalago._galago import BTreeReader  # C++ extension
    _HAS_EXTENSION = True
except ImportError:
    _HAS_EXTENSION = False
    BTreeReader = None  # type: ignore[assignment,misc]


def open_btree(path: str) -> "BTreeReader":
    """Open a Galago disk B-tree index file for reading."""
    if not _HAS_EXTENSION:
        raise RuntimeError(
            "The C++ extension (_galago) is not built. "
            "Run `pip install -e .` with a C++ compiler and CMake ≥ 3.15."
        )
    return BTreeReader(path)


__all__ = ["open_btree", "BTreeReader"]
