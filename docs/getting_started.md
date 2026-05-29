# Getting Started with PyGalago

This guide walks through the **complete workflow** for building a search engine from a raw document corpus: installation → index building → retrieval → evaluation. Every step is explained with working code. The guide is intended for both first-time users and AI agents that need to automate IR experiments.

---

## Table of Contents

1. [Prerequisites & Installation](#1-prerequisites--installation)
2. [Understanding the Data Model](#2-understanding-the-data-model)
3. [Preparing Your Corpus](#3-preparing-your-corpus)
4. [Building an Index](#4-building-an-index)
5. [Searching the Index](#5-searching-the-index)
6. [Evaluating Retrieval Quality](#6-evaluating-retrieval-quality)
7. [Running a Complete Experiment (End-to-End Script)](#7-running-a-complete-experiment-end-to-end-script)
8. [Using the CLI](#8-using-the-cli)
9. [Advanced Topics](#9-advanced-topics)
10. [Troubleshooting](#10-troubleshooting)
11. [API Quick Reference](#11-api-quick-reference)

---

## 1. Prerequisites & Installation

### System requirements

- Python 3.10 or later
- A C++17 compiler (Clang 11+ or GCC 9+)
- CMake 3.15+

These are only required at install time to compile the C++ extension. At runtime, only Python is needed.

### Install from source

```bash
git clone https://github.com/your-org/pygalago
cd pygalago
pip install -e .
```

### Optional dependencies

```bash
# Krovetz stemmer (recommended for English IR)
pip install KrovetzStemmer

# Snowball / Porter stemmer
pip install PyStemmer

# Fast TREC evaluation (alternative to built-in metrics)
pip install pytrec-eval-terrier
```

### Verify the installation

```python
import pygalago
print(pygalago.__version__)   # 0.1.0

import pygalago._galago       # C++ extension
print("C++ extension loaded successfully")

# CLI
# pygalago --version
```

---

## 2. Understanding the Data Model

### Documents

The fundamental unit is a `Document`:

```python
from pygalago.parse.document import Document, Tag

doc = Document(
    name="FBIS3-1",           # unique string identifier (required)
    text="The raw document text...",  # full text content
    terms=[],                 # filled in by tokenize()
    tags=[],                  # SGML/HTML tags with token ranges
    metadata={"url": "..."},  # arbitrary key-value pairs
    identifier=-1,            # internal docid, set by IndexBuilder
)
```

### Index

An index is a directory containing B-tree files:

| File | Maps | Used for |
|---|---|---|
| `names` | docid → name | Resolving result docids to readable names |
| `lengths` | field → `int32[]` | BM25 length normalisation |
| `postings` | term → posting list | Unstemmed term lookup |
| `postings.krovetz` | stemmed term → posting list | Default stemmed lookup |

### Internal docids

Documents are assigned consecutive integer ids (`0, 1, 2, …`) in the order they are added during index building. The `names` part maps these internal ids back to the original string identifiers.

---

## 3. Preparing Your Corpus

PyGalago can parse four collection formats. Choose the one matching your data.

### 3.1 TREC text format (`.trec`, `.txt`, `.sgml`)

The standard format for TREC ad-hoc collections (Robust04, WT10g, GOV2):

```
<DOC>
<DOCNO> FBIS3-1 </DOCNO>
<TEXT>
The Financial Times reported today that international trade...
</TEXT>
</DOC>
<DOC>
<DOCNO> FBIS3-2 </DOCNO>
<HEADLINE>
EU trade negotiations
</HEADLINE>
<TEXT>
European Union officials announced...
</TEXT>
</DOC>
```

Recognised content tags (text extracted from these):
`<TEXT>`, `<HEADLINE>`, `<TITLE>`, `<HL>`, `<HEAD>`, `<TTL>`, `<DD>`, `<DATE>`, `<LP>`, `<LEADPARA>`.

### 3.2 TREC web format (`.web`)

Used by WT10g and GOV2:

```
<DOC>
<DOCNO>WTX001-B01-1</DOCNO>
<DOCHDR>
http://www.example.com/
...
</DOCHDR>
<html>…raw HTML body…</html>
</DOC>
```

### 3.3 JSON / JSON-Lines (`.json`, `.jsonl`)

One document per line. Supported field names for id: `id`, `docid`, `docno`, `identifier`. Supported field names for text: `text`, `contents`, `body`, `passage`.

```json
{"id": "doc001", "text": "The quick brown fox..."}
{"id": "doc002", "text": "Information retrieval is..."}
```

Or a JSON array:

```json
[
  {"docno": "d1", "body": "first document"},
  {"docno": "d2", "body": "second document"}
]
```

### 3.4 WARC format (`.warc`)

Web archive format used by ClueWeb09/12:

```
WARC/1.0
WARC-Type: response
WARC-TREC-ID: clueweb09-en0000-01-00001
WARC-Target-URI: http://example.com/
Content-Length: 1234

HTTP/1.1 200 OK
Content-Type: text/html

<html>…</html>
```

### 3.5 Gzip-compressed files

Any format can be gzip-compressed; just add `.gz` to the extension:

```
collection.trec.gz
collection.jsonl.gz
```

### 3.6 Auto-detection

`open_collection()` detects the format from the file extension:

```python
from pygalago.parse.parsers import open_collection

for doc in open_collection("collection.trec"):
    print(doc.name, len(doc.text))
```

### 3.7 Creating a minimal synthetic corpus (for testing)

```python
from pygalago.parse.document import Document

docs = [
    Document.from_text("doc001", "Information retrieval is a field of study."),
    Document.from_text("doc002", "Search engines index documents using inverted lists."),
    Document.from_text("doc003", "BM25 is a popular ranking function in modern IR."),
]
```

---

## 4. Building an Index

### 4.1 Basic usage — `IndexBuilder`

```python
from pygalago.index.builder import IndexBuilder

with IndexBuilder(
    output_path="/path/to/my-index",
    stemmer="krovetz",       # "krovetz" | "porter" | "none"
    also_unstemmed=True,     # also write unstemmed `postings` part
    chunk_size=100_000,      # max documents in memory at once
) as builder:
    n = builder.add_documents_from_file("collection.trec")

print(f"Indexed {n} documents")
```

**What `with … as builder` does:** `__exit__` calls `builder.build()`, which writes all accumulated postings to disk. Without the context manager, call `builder.build()` manually.

### 4.2 Stemmer options

| `stemmer=` | Index part written | Package required |
|---|---|---|
| `"krovetz"` | `postings.krovetz` | `pip install KrovetzStemmer` |
| `"porter"` | `postings.porter` | `pip install PyStemmer` |
| `"none"` | `postings` | None |

If the stemming package is not installed, a warning is printed and the identity stemmer is used (no stemming).

**For Robust04 / most TREC collections:** use `stemmer="krovetz"` (the Java Galago default).

### 4.3 Adding documents one at a time

```python
from pygalago.parse.document import Document
from pygalago.parse.tokenizer import tokenize
from pygalago.index.builder import IndexBuilder

with IndexBuilder("/path/to/index", stemmer="krovetz") as builder:
    for i, line in enumerate(open("raw_text.txt")):
        doc = Document.from_text(f"doc{i:06d}", line.strip())
        builder.add_document(doc)   # tokenise is called internally
```

### 4.4 Adding documents from multiple files

```python
import glob

with IndexBuilder("/path/to/index", stemmer="krovetz") as builder:
    for path in sorted(glob.glob("/data/robust04/disk*/*.sgml.gz")):
        count = builder.add_documents_from_file(path)
        print(f"  {path}: {count} docs")
```

### 4.5 What happens during indexing

1. `tokenize(doc)` — strips HTML tags, splits on whitespace/punctuation, lowercases. Fills `doc.terms`.
2. Term frequencies are counted per document.
3. Each stemmed/unstemmed term's `(docid, tf)` pair is added to the in-memory accumulator.
4. On `build()`:
   a. `write_names()` — B-tree of `int64 BE docid → UTF-8 name`.
   b. `write_lengths()` — B-tree of `"document" → header + int32[] array`.
   c. `write_postings_index()` — B-tree of `term → encoded posting list` (sorted by term).
   d. `buildManifest.json` — metadata.

### 4.6 Monitoring build progress

```python
with IndexBuilder("/path/to/index", stemmer="krovetz") as builder:
    n = builder.add_documents_from_file("collection.trec")
    print(f"Indexed {builder.document_count} docs, "
          f"{builder.total_tokens} tokens")
```

### 4.7 Memory considerations

All postings accumulate in RAM until `build()` is called. For large collections:

- `chunk_size=100_000` (default) keeps ~100K docs in memory at a time.
- For Robust04 (~528K docs), the unstemmed vocabulary is ~800K terms; peak RAM usage is roughly **1–4 GB** depending on average term counts.
- For very large collections (>10M docs), consider batching with multiple `IndexBuilder` instances and merging the parts (not yet automated — planned for a future release).

---

## 5. Searching the Index

### 5.1 The `Retrieval` class (recommended)

```python
from pygalago.retrieval import Retrieval

r = Retrieval(
    index_path="/path/to/my-index",
    b=0.75,                       # BM25 length normalisation (default 0.75)
    k=1.2,                        # BM25 term saturation (default 1.2)
    part="postings.krovetz",      # which index part to use
)

results = r.search("information retrieval", n=1000)
# returns [(doc_name, score), ...]  sorted by descending score

for rank, (name, score) in enumerate(results, 1):
    print(f"{rank:4d}  {score:10.4f}  {name}")
```

**`part` must match the index**: use `"postings.krovetz"` if you built with `stemmer="krovetz"`, `"postings"` if `stemmer="none"`, etc.

### 5.2 Query language

The `search()` method accepts both plain text and structured queries:

```python
# Plain keyword query — terms combined with equal weights
r.search("international organized crime")

# Explicit #combine
r.search("#combine(information retrieval)")

# Weighted #combine (0.7 × term1 + 0.3 × term2)
r.search("#combine:0=0.7:1=0.3(information retrieval)")

# Full Dependence Model / Sequential Dependence Model
r.search("#fdm(information retrieval systems)")

# SDM with custom weights
r.search("#fdm:uniw=0.9:odw=0.07:uww=0.03(new york crime)")

# Ordered phrase (currently treated as unigrams; proximity requires positional index)
r.search("#od:1(new york)")
```

See [query_language.md](query_language.md) for the complete syntax reference.

### 5.3 BM25 parameter tuning

```python
# Standard parameters for title queries (short, specific)
r_title = Retrieval("/path/to/index", b=0.4, k=1.0)

# Standard parameters for long document queries
r_verbose = Retrieval("/path/to/index", b=0.75, k=1.2)

# Retrieve more candidates, re-rank later
results = r.search("query text", n=10_000)
```

### 5.4 Low-level retrieval (without query parsing)

```python
from pygalago.retrieval import bm25_search

# Terms must already be stemmed to match the index part
results = bm25_search(
    "/path/to/index",
    terms=["inform", "retriev"],   # Krovetz-stemmed
    b=0.75, k=1.2, n=1000,
    part="postings.krovetz",
)
for sd in results:
    print(sd.document, sd.score)   # internal docid, score
```

### 5.5 Iterating postings directly

For diagnostic purposes or building custom scoring functions:

```python
import pygalago._galago as g

pr = g.PostingsReader("/path/to/index/postings.krovetz")

# Check collection statistics
stats = pr.get_stats("inform")
if stats:
    print(f"df={stats['document_count']}  cf={stats['collection_count']}")

# Iterate all postings for a term
it = pr.get_postings("inform")
if it:
    for doc_id, tf in it:
        print(f"  doc {doc_id}: tf={tf}")
```

---

## 6. Evaluating Retrieval Quality

### 6.1 TREC qrels format

Qrels (relevance judgments) use the standard TREC 4-column format:

```
301 0 FBIS3-1 1
301 0 FBIS3-2 0
302 0 FT941-1 1
```

Columns: `topic_id  iteration  doc_id  relevance_grade`

- `iteration` is conventionally `0`.
- `relevance_grade ≥ 1` is relevant; `0` is not relevant.
- Graded relevance (`0/1/2/3`) is supported for NDCG.

### 6.2 Running evaluation

```python
from pygalago.eval import read_qrels, read_run, evaluate

# Load qrels
qrels = read_qrels("qrels.robust04.txt")

# Load a TREC run file
run = read_run("my_run.txt")

# Convert run to ranked lists
ranked = {topic: [rd.doc_id for rd in docs]
          for topic, docs in run.items()}

# Compute all standard metrics
scores = evaluate(ranked, qrels,
    metrics=["map", "ndcg@10", "ndcg@20", "p@5", "p@10", "mrr", "bpref"])

for metric, value in scores.items():
    print(f"{metric:<15}  {value:.4f}")
```

### 6.3 Per-topic evaluation

```python
from pygalago.eval.metrics import average_precision
from pygalago.eval.qrels   import relevant_docs

for topic, docs in sorted(ranked.items()):
    rel = relevant_docs(qrels, topic)
    ap  = average_precision(docs, rel)
    print(f"Topic {topic}: AP={ap:.4f}  rel={len(rel)}")
```

### 6.4 Writing a TREC run file from search results

```python
from pygalago.retrieval import Retrieval
from pygalago.eval.run  import write_run, Run, RankedDoc

r = Retrieval("/path/to/index")
run: Run = {}

topics = {
    "301": "international organized crime",
    "302": "poliomyelitis eradication",
    "303": "hubble telescope repair",
}

for topic_id, query in topics.items():
    results = r.search(query, n=1000)
    run[topic_id] = [
        RankedDoc(doc_id=name, score=score, rank=i)
        for i, (name, score) in enumerate(results, 1)
    ]

write_run(run, "my_run.txt", run_tag="pygalago_bm25")
```

### 6.5 Available metrics

| Metric name | Description |
|---|---|
| `"map"` | Mean Average Precision |
| `"ndcg@k"` (e.g. `"ndcg@10"`) | Normalised Discounted Cumulative Gain at rank k |
| `"p@k"` (e.g. `"p@10"`) | Precision at rank k |
| `"mrr"` | Mean Reciprocal Rank |
| `"bpref"` | Binary Preference (Buckley & Voorhees 2004) |
| `"r-prec"` | R-Precision |

---

## 7. Running a Complete Experiment (End-to-End Script)

The following self-contained script performs a full ad-hoc retrieval experiment on any TREC collection.

```python
#!/usr/bin/env python3
"""
End-to-end PyGalago experiment.

Usage:
    python run_experiment.py \
        --collection /data/robust04/*.sgml.gz \
        --queries    topics.301-450.txt \
        --qrels      qrels.robust04.txt \
        --index      /tmp/robust04.index \
        --output     run.txt \
        --stemmer    krovetz \
        -n           1000
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time

from pygalago.index.builder import IndexBuilder
from pygalago.retrieval      import Retrieval
from pygalago.eval           import read_qrels, evaluate
from pygalago.eval.run       import write_run, Run, RankedDoc


def parse_topic_file(path: str) -> dict[str, str]:
    """Parse a TREC topic file (title field only).

    Handles both old-style TREC-format topics and simple tab-separated files.
    """
    topics: dict[str, str] = {}

    with open(path, encoding="utf-8") as f:
        content = f.read()

    # Simple tab-separated: "topic_id\tquery text"
    if "\t" in content.splitlines()[0]:
        for line in content.splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    topics[parts[0].strip()] = parts[1].strip()
        return topics

    # TREC format: <top> <num> <title> blocks
    current_topic = None
    in_title = False
    buf: list[str] = []

    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith("<num>"):
            current_topic = stripped.split("Number:", 1)[-1].strip() \
                              if "Number:" in stripped else \
                              stripped.replace("<num>", "").replace("</num>", "").strip()
            in_title = False
        elif "<title>" in stripped.lower():
            in_title = True
            rest = stripped.split(">", 1)[-1].strip() if ">" in stripped else ""
            if rest and "</title>" not in rest.lower():
                buf = [rest]
        elif in_title:
            if "</title>" in stripped.lower() or stripped.startswith("<"):
                in_title = False
                if current_topic and buf:
                    topics[current_topic] = " ".join(buf).strip()
                buf = []
            else:
                buf.append(stripped)

    return topics


def build_index_if_needed(
    collection_paths: list[str],
    index_path: str,
    stemmer: str,
) -> None:
    if os.path.isfile(os.path.join(index_path, "buildManifest.json")):
        print(f"Index already exists at {index_path!r} — skipping build.")
        return

    print(f"Building index → {index_path!r}  (stemmer={stemmer})")
    t0 = time.perf_counter()
    total = 0

    with IndexBuilder(index_path, stemmer=stemmer, also_unstemmed=False) as b:
        for path in collection_paths:
            n = b.add_documents_from_file(path)
            total += n
            print(f"  {os.path.basename(path)}: +{n} docs  (total={total})")

    print(f"Build complete: {total} docs in {time.perf_counter()-t0:.1f}s")


def run_retrieval(
    index_path: str,
    topics: dict[str, str],
    stemmer: str,
    n: int,
    b: float,
    k: float,
) -> Run:
    part = f"postings.{stemmer}" if stemmer != "none" else "postings"
    r = Retrieval(index_path, b=b, k=k, part=part)

    run: Run = {}
    for topic_id, query in sorted(topics.items()):
        results = r.search(query, n=n)
        run[topic_id] = [
            RankedDoc(doc_id=name, score=score, rank=i)
            for i, (name, score) in enumerate(results, 1)
        ]

    return run


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--collection", nargs="+", required=True)
    ap.add_argument("--queries",    required=True)
    ap.add_argument("--qrels",      required=True)
    ap.add_argument("--index",      required=True)
    ap.add_argument("--output",     required=True)
    ap.add_argument("--stemmer",    default="krovetz",
                    choices=["krovetz", "porter", "none"])
    ap.add_argument("-n",           type=int, default=1000)
    ap.add_argument("-b",           type=float, default=0.75)
    ap.add_argument("-k",           type=float, default=1.2)
    args = ap.parse_args()

    # Expand glob patterns
    collection_paths: list[str] = []
    for pat in args.collection:
        expanded = sorted(glob.glob(pat))
        if not expanded:
            print(f"Warning: no files matched {pat!r}", file=sys.stderr)
        collection_paths.extend(expanded)

    if not collection_paths:
        sys.exit("Error: no collection files found")

    # 1. Build index
    build_index_if_needed(collection_paths, args.index, args.stemmer)

    # 2. Parse topics
    topics = parse_topic_file(args.queries)
    print(f"Loaded {len(topics)} topics from {args.queries!r}")

    # 3. Retrieve
    print(f"Running retrieval (n={args.n}, b={args.b}, k={args.k})…")
    t0 = time.perf_counter()
    run = run_retrieval(args.index, topics, args.stemmer, args.n, args.b, args.k)
    print(f"Retrieval complete: {sum(len(v) for v in run.values())} result rows "
          f"in {time.perf_counter()-t0:.1f}s")

    # 4. Write run file
    write_run(run, args.output, run_tag="pygalago_bm25")
    print(f"Run written to {args.output!r}")

    # 5. Evaluate
    qrels = read_qrels(args.qrels)
    ranked = {t: [rd.doc_id for rd in docs] for t, docs in run.items()}
    scores = evaluate(ranked, qrels,
                      metrics=["map", "ndcg@10", "ndcg@20", "p@10", "mrr", "bpref"])

    print("\n" + "="*40)
    print("Results")
    print("="*40)
    for metric, value in scores.items():
        print(f"  {metric:<15}  {value:.4f}")


if __name__ == "__main__":
    main()
```

**Example invocation for Robust04:**

```bash
python run_experiment.py \
    --collection "/data/robust04/disk4/fbis/*.sgml.gz" \
                 "/data/robust04/disk5/latimes/*.gz" \
    --queries    topics.301-450.txt \
    --qrels      qrels.robust04.txt \
    --index      /tmp/robust04.index \
    --output     robust04_bm25.txt \
    --stemmer    krovetz \
    -n           1000
```

---

## 8. Using the CLI

All operations above have CLI equivalents.

### Build an index

```bash
pygalago build-index collection.trec \
    --index /path/to/index \
    --stemmer krovetz
```

Options:
- `--stemmer krovetz|porter|none` — stemming algorithm (default: `krovetz`)
- `--no-unstemmed` — skip writing the bare `postings` part
- `--chunk-size N` — documents per memory chunk (default: 100000)

### Search interactively

```bash
pygalago search --index /path/to/index \
    --query "information retrieval" \
    --count 10 \
    --stemmer krovetz
```

### Batch retrieval

```bash
# queries.tsv: one "topic_id<TAB>query" per line
pygalago batch-search \
    --index   /path/to/index \
    --queries queries.tsv \
    --output  run.txt \
    -n        1000 \
    --stemmer krovetz \
    --run-tag my_experiment
```

### Evaluate

```bash
pygalago eval \
    --qrels   qrels.txt \
    --results run.txt \
    --metrics map ndcg@10 p@10 mrr bpref \
    --per-topic
```

### Inspect index internals

```bash
# Show all terms in the postings part (first 20)
pygalago dump-index --index /path/to/index \
    --part postings.krovetz --max-terms 20

# Show all document names
pygalago dump-index --index /path/to/index --part names
```

---

## 9. Advanced Topics

### 9.1 Writing a custom parser

If your corpus has a non-standard format, implement a generator over `Document` objects:

```python
from pygalago.parse.document import Document

def parse_my_format(path: str):
    with open(path) as f:
        for line in f:
            fields = line.strip().split("|")
            if len(fields) >= 2:
                yield Document.from_text(fields[0], fields[1])

from pygalago.index.builder import IndexBuilder

with IndexBuilder("/path/to/index", stemmer="krovetz") as b:
    for doc in parse_my_format("my_corpus.psv"):
        b.add_document(doc)
```

### 9.2 Custom tokenisation

Replace the default tokeniser by pre-populating `doc.terms`:

```python
import spacy
nlp = spacy.load("en_core_web_sm")

def tokenize_with_spacy(doc):
    parsed = nlp(doc.text)
    doc.terms = [t.lemma_.lower() for t in parsed
                 if not t.is_stop and not t.is_punct]
    return doc

from pygalago.index.builder import IndexBuilder

with IndexBuilder("/path/to/index", stemmer="none") as b:
    for raw_doc in open_collection("collection.trec"):
        tokenize_with_spacy(raw_doc)
        b.add_document(raw_doc)
```

### 9.3 Reading an existing Java Galago index

PyGalago reads indexes built by Java Galago 3.x without any conversion:

```python
from pygalago.retrieval import Retrieval

# Java Galago index directory
r = Retrieval("/path/to/java-galago-index")
results = r.search("information retrieval", n=1000)
```

### 9.4 Re-ranking with a custom scorer

Use low-level access to post-process BM25 results:

```python
import pygalago._galago as g

idx     = g.DiskIndex("/path/to/index")
pr      = g.PostingsReader("/path/to/index/postings.krovetz")
lengths = g.LengthsSource("/path/to/index/lengths")

def custom_score(doc_id: int, query_terms: list[str]) -> float:
    dl = lengths.length(doc_id)
    score = 0.0
    for term in query_terms:
        it = pr.get_postings(term)
        if it:
            it.skip_to(doc_id)
            if not it.is_done and it.doc_id == doc_id:
                score += it.count / (dl + 50)  # simple TF normalisation
    return score

# Get BM25 candidates first, then re-rank
from pygalago.retrieval import Retrieval
r = Retrieval("/path/to/index")
candidates = r.search("information retrieval", n=1000)

reranked = sorted(
    [(name, custom_score(i, ["inform", "retriev"]))
     for i, (name, _) in enumerate(candidates)],
    key=lambda x: -x[1],
)
```

### 9.5 Parallel batch search

```python
from concurrent.futures import ProcessPoolExecutor
from pygalago.retrieval import Retrieval
from pygalago.eval.run  import write_run, Run, RankedDoc

INDEX_PATH = "/path/to/index"

def search_topic(args):
    topic_id, query = args
    r = Retrieval(INDEX_PATH)   # each process creates its own reader
    results = r.search(query, n=1000)
    return topic_id, [RankedDoc(name, score, i)
                      for i, (name, score) in enumerate(results, 1)]

topics = {
    "301": "international organized crime",
    "302": "poliomyelitis eradication",
}

run: Run = {}
with ProcessPoolExecutor(max_workers=4) as ex:
    for topic_id, ranked in ex.map(search_topic, topics.items()):
        run[topic_id] = ranked

write_run(run, "run.txt")
```

### 9.6 Programmatic BM25 parameter sweep

```python
from pygalago.retrieval import Retrieval
from pygalago.eval       import read_qrels, evaluate

qrels = read_qrels("qrels.txt")
topics = {"301": "crime", "302": "disease"}   # your topics

best_map, best_params = 0.0, {}

for b in [0.4, 0.6, 0.75, 0.9]:
    for k in [0.6, 1.0, 1.2, 1.5]:
        r = Retrieval("/path/to/index", b=b, k=k)
        ranked = {t: [n for n, _ in r.search(q, n=1000)]
                  for t, q in topics.items()}
        scores = evaluate(ranked, qrels, metrics=["map"])
        m = scores["map"]
        if m > best_map:
            best_map, best_params = m, {"b": b, "k": k}
        print(f"b={b}  k={k}  MAP={m:.4f}")

print(f"\nBest: MAP={best_map:.4f}  params={best_params}")
```

---

## 10. Troubleshooting

### `RuntimeError: The C++ extension (_galago) is not built`

```bash
pip install -e .       # rebuilds the extension
# or
pip install pygalago   # install pre-built wheel if available
```

### `ImportError: No module named 'krovetzstemmer'`

```bash
pip install KrovetzStemmer
```

The stemmer falls back to the identity function if not installed. To verify:

```python
from pygalago.parse.stemmer import get_stemmer
stem = get_stemmer("krovetz")
print(stem("information"))  # "inform" if Krovetz is installed, else "information"
```

### Index part not found

```
RuntimeError: Postings part 'postings.krovetz' not found in index at '/path/to/index'
```

Check which parts exist:

```bash
ls /path/to/index/
pygalago dump-index --index /path/to/index --part postings.krovetz
```

Then pass the correct `part=` to `Retrieval()`.

### Query returns no results

1. Verify the term is in the index: `pr.get_postings("term")` returns non-None.
2. Check that you are using the right stemmer: if the index was built with Krovetz, search with the Krovetz-stemmed form.
3. Check that `part=` matches the index part name.

```python
import pygalago._galago as g
pr = g.PostingsReader("/path/to/index/postings.krovetz")
print(pr.get_stats("inform"))    # None if term not in index
print(pr.get_stats("information"))  # None too — must use stemmed form
```

---

## 11. API Quick Reference

### Core classes

| Class | Module | Description |
|---|---|---|
| `Document` | `pygalago.parse.document` | Document data model |
| `IndexBuilder` | `pygalago.index.builder` | Index construction |
| `Retrieval` | `pygalago.retrieval` | Full search pipeline |
| `Node` | `pygalago.query.node` | Query tree node |

### Key functions

| Function | Module | Description |
|---|---|---|
| `open_collection(path)` | `pygalago.parse` | Parse a collection file (auto-detect format) |
| `tokenize(doc)` | `pygalago.parse` | Tokenise `doc.text` → `doc.terms` (in-place) |
| `tokenize_string(text)` | `pygalago.parse` | Tokenise a plain string → `[str]` |
| `get_stemmer(name)` | `pygalago.parse` | Get a stemmer function |
| `parse(query)` | `pygalago.query.parser` | Parse a query string → `Node` |
| `read_qrels(path)` | `pygalago.eval` | Load TREC qrels |
| `read_run(path)` | `pygalago.eval` | Load a TREC run file |
| `write_run(run, path)` | `pygalago.eval` | Write a TREC run file |
| `evaluate(ranked, qrels)` | `pygalago.eval` | Compute IR metrics |

### C++ extension (low-level)

| Class | Description |
|---|---|
| `_galago.DiskIndex` | Open an index directory |
| `_galago.NamesReader` | Read the `names` part |
| `_galago.LengthsReader` | Read the `lengths` part |
| `_galago.PostingsReader` | Read a `postings.*` part |
| `_galago.LengthsSource` | RAM-preloaded lengths (fast random access) |
| `_galago.BTreeReader` | Raw B-tree key-value reader |
| `_galago.PostingsIterator` | Iterate over a posting list |
| `_galago.LengthStats` | Collection length statistics |
| `_galago.ScoredDocument` | Result from `bm25_search` (`.document`, `.score`) |

See [index_format.md](index_format.md) for the binary layout of each part.
See [query_language.md](query_language.md) for the full query syntax reference.
