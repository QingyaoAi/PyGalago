# BSD License (http://www.galagosearch.org/license)
"""Stemmer interface with optional Krovetz and Snowball backends.

Priority:
  1. ``krovetz`` — KrovetzStemmer package (pip install KrovetzStemmer)
  2. ``porter``  — PyStemmer package, Porter2 algorithm (pip install PyStemmer)
  3. ``none``    — identity (no stemming)

Usage::

    from pygalago.parse.stemmer import get_stemmer
    stem = get_stemmer("krovetz")
    tokens = [stem(t) for t in ["information", "retrieval"]]
    # ["inform", "retriev"]  (Krovetz stems)
"""

from __future__ import annotations

from typing import Callable, List


def _identity_stemmer(term: str) -> str:
    return term


def _make_krovetz() -> Callable[[str], str]:
    try:
        from krovetzstemmer import Stemmer as KStemmer  # type: ignore
        ks = KStemmer()
        return lambda t: ks.stem(t)
    except ImportError:
        try:
            # Alternative package name
            import KrovetzStemmer as ksmod  # type: ignore
            ks = ksmod.KrovetzStemmer()
            return lambda t: ks.stem(t)
        except ImportError:
            return None  # type: ignore


def _make_porter() -> Callable[[str], str]:
    try:
        import Stemmer as pystemmer  # type: ignore
        s = pystemmer.Stemmer("english")
        return s.stemWord
    except ImportError:
        return None  # type: ignore


def get_stemmer(name: str = "none") -> Callable[[str], str]:
    """Return a single-word stemmer function.

    Parameters
    ----------
    name:
        ``"krovetz"`` | ``"porter"`` | ``"none"`` (default).
    """
    if name == "krovetz":
        fn = _make_krovetz()
        if fn is None:
            import warnings
            warnings.warn(
                "KrovetzStemmer package not found; falling back to identity stemmer. "
                "pip install KrovetzStemmer",
                ImportWarning,
                stacklevel=2,
            )
            return _identity_stemmer
        return fn

    if name in ("porter", "porter2", "snowball"):
        fn = _make_porter()
        if fn is None:
            import warnings
            warnings.warn(
                "PyStemmer package not found; falling back to identity stemmer. "
                "pip install PyStemmer",
                ImportWarning,
                stacklevel=2,
            )
            return _identity_stemmer
        return fn

    return _identity_stemmer


def stem_terms(terms: List[str], stemmer_name: str = "none") -> List[str]:
    """Stem a list of tokens with the named stemmer."""
    stem = get_stemmer(stemmer_name)
    return [stem(t) for t in terms]
