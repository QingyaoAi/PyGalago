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


_VOWELS = frozenset("aeiou")


def _r1_start(word: str) -> int:
    """Position where R1 starts in *word* (minimum position 3)."""
    n = len(word)
    for i in range(1, n):
        if word[i] not in _VOWELS and word[i - 1] in _VOWELS:
            return max(3, i + 1)
    return n


def _r2_start(word: str) -> int:
    """Position where R2 starts in *word*."""
    r1 = _r1_start(word)
    region = word[r1:]
    for i in range(1, len(region)):
        if region[i] not in _VOWELS and region[i - 1] in _VOWELS:
            return r1 + i + 1
    return len(word)


def _fix_ate_deletion(stem: str) -> str:
    """Apply Porter2 step-4 '-ate' deletion that PyStemmer incorrectly skips.

    PyStemmer applies step-5 '-e' deletion before step-4 '-ate' deletion for
    some words (e.g. "international" → "internat" instead of "intern").
    This function corrects that by checking whether the '-ate' form would have
    been in R2 under the strict Galago Java rule (R2 start strictly less than
    suffix start position).
    """
    if not stem.endswith("at") or len(stem) < 4:
        return stem
    candidate_ate = stem + "e"                     # reconstruct the pre-step-5 form
    # Position where "ate" starts in candidate_ate (= len - 3)
    ate_start = len(candidate_ate) - 3
    r2 = _r2_start(candidate_ate)
    # Galago Java Porter2 uses strict inequality: suffix must START AFTER R2
    if ate_start > r2:
        return stem[:-2]                            # delete "at" → step-4 correction
    return stem


def _make_porter() -> Callable[[str], str]:
    try:
        import Stemmer as pystemmer  # type: ignore
        s = pystemmer.Stemmer("english")
        base = s.stemWord

        def _galago_compat_porter(word: str) -> str:
            return _fix_ate_deletion(base(word))

        return _galago_compat_porter
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
