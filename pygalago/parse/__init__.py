"""pygalago.parse — document parsing and tokenisation (Phase 5)."""

from pygalago.parse.document  import Document, Tag
from pygalago.parse.tokenizer import tokenize, tokenize_string
from pygalago.parse.stemmer   import get_stemmer, stem_terms
from pygalago.parse.parsers   import open_collection

__all__ = [
    "Document", "Tag",
    "tokenize", "tokenize_string",
    "get_stemmer", "stem_terms",
    "open_collection",
]
