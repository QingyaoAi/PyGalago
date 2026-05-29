"""PyGalago — Python + C++ port of the Galago search engine."""

__version__ = "0.1.0"

# Lazy-import sub-packages so `import pygalago` is fast even without the C++
# extension built yet.
from pygalago.parameters import Parameters

__all__ = ["Parameters"]
