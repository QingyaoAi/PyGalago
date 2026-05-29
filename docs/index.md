# PyGalago Documentation

**PyGalago** is a Python + C++ search engine — a complete rewrite of Galago 3.22 that requires no Java runtime.

## Documents

| Document | Contents |
|---|---|
| [Getting Started](getting_started.md) | Full walkthrough from raw corpus to evaluated results; end-to-end script; CLI reference; troubleshooting |
| [Query Language](query_language.md) | GSQL syntax, all operators (`#combine`, `#fdm`, `#od`, `#uw`, …), BM25 formula, Python query-tree API |
| [Index Format](index_format.md) | Binary layout of every index part; B-tree structure; VByte encoding; C++ types accessible from Python |
| [Migration Guide](migration.md) | Java Galago → PyGalago mapping for every major operation |

## Quick links

- **Install:** `pip install pygalago`
- **Build an index:** `pygalago build-index collection.trec --index /path/to/index --stemmer krovetz`
- **Search:** `pygalago search --index /path/to/index --query "information retrieval"`
- **Evaluate:** `pygalago eval --qrels qrels.txt --results run.txt`
- **Python API:** see [Getting Started §11](getting_started.md#11-api-quick-reference)
