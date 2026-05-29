# Migrating from Java Galago to PyGalago

This guide maps the most common Java Galago operations to their PyGalago equivalents.

---

## 1. Installation

| Java Galago | PyGalago |
|---|---|
| Download JAR + scripts | `pip install pygalago` |
| `java -jar galago.jar` | `pygalago` CLI |
| Set up Maven / classpath | Nothing extra |

---

## 2. Building an Index

### Java

```bash
galago build-index --indexPath=/path/to/index \
    --inputPath=/path/to/collection \
    --stemmer+krovetz
```

### PyGalago CLI

```bash
pygalago build-index collection.trec \
    --index /path/to/index \
    --stemmer krovetz
```

### PyGalago Python API

```python
from pygalago.index.builder import IndexBuilder

with IndexBuilder("/path/to/index", stemmer="krovetz") as b:
    b.add_documents_from_file("collection.trec")
    # or add documents one at a time:
    # b.add_document(doc)
```

**Index parts produced** — binary-compatible with Java Galago:

| Part | Description |
|---|---|
| `names` | docid → document name |
| `lengths` | docid → token count |
| `postings` | term → posting list (unstemmed) |
| `postings.krovetz` | term → posting list (Krovetz-stemmed) |
| `buildManifest.json` | build metadata |

You can open a PyGalago-built index with Java Galago and vice versa.

---

## 3. Searching

### Java (LocalRetrieval)

```java
LocalRetrieval retrieval = new LocalRetrieval("/path/to/index");
Node query = StructuredQuery.parse("information retrieval");
Parameters qp = new Parameters();
qp.set("requested", 1000);
List<ScoredDocument> results = retrieval.executeQuery(query, qp).scoredDocuments;
```

### PyGalago

```python
from pygalago.retrieval import Retrieval

r = Retrieval("/path/to/index")
results = r.search("information retrieval", n=1000)
# [("FBIS3-1", -4.21), ("FT931-1", -4.35), ...]
```

Structured query syntax is the same as Java Galago:

```python
r.search("#combine(information retrieval)")
r.search("#combine:0=0.8:1=0.2(information retrieval)")
r.search("#od:1(new york)")
r.search("#uw:8(search engine)")
```

### Batch search

**Java:**
```bash
galago batch-search --index=/path/to/index \
    --query=queries.json --outputFile=run.txt
```

**PyGalago:**
```bash
pygalago batch-search --index /path/to/index \
    --queries queries.tsv --output run.txt
```

Query file format (tab-separated, one query per line):
```
301   international organized crime
302   poliomyelitis eradication
303   hubble telescope repair
```

---

## 4. Reading Index Parts Directly

### Java

```java
DiskIndex index = new DiskIndex("/path/to/index");
int docLen = index.getLength(docid);
String docName = index.getName(docid);
NodeStatistics stats = index.getCollectionStatistics("#counts:information");
```

### PyGalago

```python
import pygalago._galago as g

idx = g.DiskIndex("/path/to/index")
doc_len  = idx.get_length(0)
doc_name = idx.get_name(0)
stats    = idx.get_length_stats()  # avg_length, collection_length, …

# Postings iterator
pr = g.PostingsReader("/path/to/index/postings.krovetz")
it = pr.get_postings("information")
while not it.is_done:
    print(it.doc_id, it.count)
    it.next()
```

---

## 5. Evaluation

### Java

```bash
galago eval --baseline=run.txt --judgments=qrels.txt --metrics+map --metrics+ndcg10
```

### PyGalago CLI

```bash
pygalago eval --qrels qrels.txt --results run.txt \
    --metrics map ndcg@10 p@10 mrr bpref
```

### PyGalago Python API

```python
from pygalago.eval import read_qrels, read_run, evaluate

qrels = read_qrels("qrels.txt")
run   = read_run("run.txt")
ranked = {t: [r.doc_id for r in docs] for t, docs in run.items()}

scores = evaluate(ranked, qrels,
    metrics=["map", "ndcg@10", "ndcg@20", "p@10", "mrr", "bpref"])

for metric, score in scores.items():
    print(f"{metric:<15} {score:.4f}")
```

Individual metrics are also available:

```python
from pygalago.eval.metrics import (
    average_precision, ndcg_at_k, reciprocal_rank,
    precision_at_k, bpref
)
```

---

## 6. Query Language Mapping

| Java Galago operator | PyGalago equivalent | Notes |
|---|---|---|
| `#combine(a b c)` | `#combine(a b c)` | Uniform weight combination |
| `#combine:0=0.8:1=0.2(a b)` | `#combine:0=0.8:1=0.2(a b)` | Weighted combination |
| `#od:2(a b)` | `#od:2(a b)` | Ordered window |
| `#uw:4(a b)` | `#uw:4(a b)` | Unordered window |
| `#sdm(a b c)` | `#sdm(a b c)` | Sequential dependency model |
| `#counts:term()` | plain `term` | Leaf term node |
| `#text:term()` | plain `term` | Leaf term node |

---

## 7. Stemming

| Java Galago | PyGalago |
|---|---|
| `--stemmer+krovetz` | `--stemmer krovetz` |
| `--stemmer+porter` | `--stemmer porter` (requires `pip install PyStemmer`) |
| No stemming | `--stemmer none` |

```python
from pygalago.parse.stemmer import get_stemmer

stem = get_stemmer("krovetz")
print(stem("running"))    # → "run"
print(stem("information")) # → "inform"
```

---

## 8. Document Parsing

### Java

```java
DocumentReader reader = DocumentReader.Instance(params);
// custom reader implementations per format
```

### PyGalago

```python
from pygalago.parse import open_collection, tokenize

for doc in open_collection("collection.trec"):
    tokenize(doc)
    print(doc.name, len(doc.terms), "tokens")
```

Supported formats are auto-detected from the file extension:
`.trec`, `.web`, `.json`, `.jsonl`, `.warc`, and `.gz` variants.

---

## 9. Feature Differences

| Feature | Java Galago | PyGalago |
|---|---|---|
| BM25 retrieval | ✓ | ✓ |
| Language model retrieval | ✓ | Planned (Phase 4+) |
| SDM / FDM expansion | ✓ | ✓ |
| TREC text / web / WARC parsing | ✓ | ✓ |
| Krovetz stemming | ✓ | ✓ (via KrovetzStemmer package) |
| Porter/Snowball stemming | ✓ | ✓ (via PyStemmer package) |
| Galago-format index (read) | ✓ | ✓ |
| Galago-format index (write) | ✓ | ✓ |
| MAP, NDCG, MRR, Bpref | ✓ | ✓ |
| TupleFlow distributed indexing | ✓ | Not ported (use multiprocessing) |
| Positional / proximity operators | ✓ | Planned |
| Learning-to-rank features | ✓ | Planned |
| Anchor text / links index | ✓ | Not ported |
| Galago web UI | ✓ | Not ported |

---

## 10. Index Compatibility

PyGalago reads and writes indexes in the same binary format as Java Galago 3.x.
An index built with one can be read by the other without conversion.

The only exception is the `corpus` part (raw document store), which is not yet
implemented in PyGalago. All other parts (`names`, `lengths`, `postings.*`) are
fully compatible.
