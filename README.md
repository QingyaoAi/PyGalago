# PyGalago

Python + C++ port of the [Galago 3.22](http://sourceforge.net/projects/lemur/) search engine.
No Java runtime required — `pip install pygalago` is enough.

## Why?

| | Java Galago | PyGalago |
|---|---|---|
| Runtime | JVM required | Python 3.10+ |
| Integration | External process / JNI | `import pygalago` |
| Performance-critical paths | JIT-compiled | C++17 via pybind11 |
| Scripting | Subprocess calls | Pure Python API |
| Distribution | JAR + scripts | `pip install pygalago` |

## Quick start

```bash
pip install pygalago
```

### Build an index

```bash
pygalago build-index collection.trec --index /path/to/my-index --stemmer krovetz
```

Or from Python:

```python
from pygalago.index.builder import IndexBuilder

with IndexBuilder("/path/to/my-index", stemmer="krovetz") as b:
    b.add_documents_from_file("collection.trec")
```

### Search

```bash
# Single query
pygalago search --index /path/to/my-index --query "information retrieval" -n 10

# Batch run file
pygalago batch-search --index /path/to/my-index \
    --queries topics.tsv --output run.txt -n 1000
```

Or from Python:

```python
from pygalago.retrieval import Retrieval

r = Retrieval("/path/to/my-index")
results = r.search("information retrieval", n=1000)
# [("FBIS3-1", -4.21), ("FT931-1", -4.35), ...]
```

### Evaluate

```bash
pygalago eval --qrels qrels.robust04.txt --results run.txt \
    --metrics map ndcg@10 p@10 mrr bpref
```

Or from Python:

```python
from pygalago.eval import read_qrels, read_run, evaluate

qrels = read_qrels("qrels.robust04.txt")
run   = read_run("run.txt")
ranked = {t: [r.doc_id for r in docs] for t, docs in run.items()}
scores = evaluate(ranked, qrels)
print(scores)
# {'map': 0.254, 'ndcg@10': 0.421, 'p@10': 0.382, 'mrr': 0.623, 'bpref': 0.267}
```

## CLI reference

| Command | Description |
|---|---|
| `pygalago build-index <collection> --index DIR` | Build a Galago-format index |
| `pygalago search --index DIR --query TEXT` | Run a single query |
| `pygalago batch-search --index DIR --queries FILE --output FILE` | Run a query file |
| `pygalago dump-index --index DIR --part PART` | Inspect index B-tree parts |
| `pygalago eval --qrels FILE --results FILE` | Score a TREC run |

## Supported collection formats

| Extension | Parser |
|---|---|
| `.trec`, `.txt`, `.sgml` | TREC text (`<DOC>…<DOCNO>…<TEXT>…</DOC>`) |
| `.web` | TREC web / GOV2 format |
| `.json`, `.jsonl` | JSON or JSON-Lines |
| `.warc` | WARC/1.0–1.1 web archives |
| `.gz` | Gzip-compressed variant of any of the above |

## Python API

```python
# Parse documents
from pygalago.parse import Document, open_collection, tokenize, get_stemmer

for doc in open_collection("collection.trec"):
    tokenize(doc)             # fills doc.terms
    print(doc.name, doc.terms[:5])

# Build index
from pygalago.index.builder import IndexBuilder
with IndexBuilder("my-index", stemmer="krovetz") as b:
    b.add_documents_from_file("collection.trec")

# Search
from pygalago.retrieval import Retrieval
r = Retrieval("my-index")
results = r.search("#combine(information retrieval)", n=100)

# Evaluate
from pygalago.eval import read_qrels, evaluate
qrels = read_qrels("qrels.txt")
scores = evaluate({"q1": ["doc1", "doc2"]}, qrels)
```

## Building from source

```bash
git clone https://github.com/your-org/pygalago
cd pygalago
pip install -e ".[dev,stemming]"
pytest
```

Requirements: Python ≥ 3.10, CMake ≥ 3.15, a C++17 compiler, pybind11.

## Optional dependencies

| Package | Purpose |
|---|---|
| `PyStemmer` | Porter/Snowball stemming |
| `KrovetzStemmer` | Krovetz stemming (falls back to identity if absent) |
| `pytrec-eval-terrier` | Fast trec_eval wrapper (alternative to built-in metrics) |

## Benchmarks (Apple M-series, Robust04 — 528K docs)

| Operation | Throughput / Latency |
|---|---|
| Index build | ~80 K docs/s |
| Single-term BM25 (top-1000) | ~50 ms |
| Multi-term BM25 (top-1000) | ~150 ms |

Run `python scripts/benchmark.py --index /path/to/index` to reproduce.

## License

BSD License. See [LICENSE](LICENSE) for details.

Original Galago: © University of Massachusetts Amherst, CIIR.
