"""pytest configuration — auto-discovers local Galago indexes for integration tests.

This file is committed to git; the index directories themselves are not (they are
listed in .gitignore under index-*/).

Priority order for finding the Robust04 index:
  1. GALAGO_INDEX_PATH env var (set by CI or the user at runtime)
  2. KNOWN_INDEXES list below (machine-specific paths that exist locally)

To add another local index location, append to KNOWN_INDEXES.  The first
path that exists on disk wins and is automatically exposed as GALAGO_INDEX_PATH
so every test that reads that variable picks it up without any manual setup.
"""

import os

# ── Known local index paths (ordered by preference) ──────────────────────────
# These are absolute paths on developer machines.  Missing paths are silently
# skipped — they only activate when the directory exists locally.
KNOWN_INDEXES = [
    "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04.index",
]

# ── Auto-discover ─────────────────────────────────────────────────────────────


def _find_index() -> str:
    # Explicit env var takes priority — allows CI and one-off overrides.
    env = os.environ.get("GALAGO_INDEX_PATH", "")
    if env and os.path.isdir(env):
        return env

    for path in KNOWN_INDEXES:
        if os.path.isdir(path):
            return path

    return ""


_discovered = _find_index()
if _discovered:
    os.environ.setdefault("GALAGO_INDEX_PATH", _discovered)
