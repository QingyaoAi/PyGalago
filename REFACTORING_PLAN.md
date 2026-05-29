# PyGalago Refactoring Plan: Java → Python + C++

**Date:** 2026-05-29  
**Scope:** Complete rewrite of Galago 3.22 (~131K lines Java) in Python + C++  
**Goal:** Eliminate the Java runtime dependency while preserving all search engine functionality  

---

## 1. Why Python + C++?

| Concern | Java today | Python + C++ target |
|---------|-----------|---------------------|
| Runtime dependency | JVM required | Python 3.10+ only |
| Native integration | JNI friction | pybind11 for zero-copy interop |
| Performance-critical paths | JIT-compiled | C++ (equal or faster cold start) |
| Research scripting | External process | `import pygalago` |
| Package distribution | JAR + scripts | `pip install pygalago` |

---

## 2. Codebase Inventory

| Module | Lines (Java) | Role |
|--------|-------------|------|
| `core` | 65,415 | Search engine — retrieval, indexing, parsing |
| `snowball-stemmers` | 22,322 | 15-language Snowball/Porter2 stemmers |
| `contrib` | 14,420 | Experimental retrievers, parsers, LTR features |
| `tupleflow` | 10,852 | Distributed/parallel batch-processing framework |
| `krovetz-stemmer` | 7,392 | Krovetz stemmer |
| `utility` | 6,502 | B-tree, buffers, JSON `Parameters`, compression |
| `eval` | 3,714 | IR metrics (MAP, NDCG, P@k, MRR, …) |
| `tupleflow-typebuilder` | 710 | Maven code-generator for TupleFlow types |
| **Total** | **131,327** | |

---

## 3. Architecture Decision

```
┌────────────────────────────────────────────────┐
│              Python Public API                 │
│  pygalago.{index, retrieval, eval, tools}      │
│  (query DSL, CLI wrappers, batch scripts)      │
├────────────────────────────────────────────────┤
│           pybind11 Binding Layer               │
│  thin C++ → Python bridge; zero-copy buffers   │
├────────────────────────────────────────────────┤
│              C++ Core Library (libgalago)      │
│  index I/O · inverted-list iterators           │
│  BM25/LM scoring · proximity operators        │
│  compression · tokenizer                       │
└────────────────────────────────────────────────┘
```

### Language split rationale

| Subsystem | Language | Reason |
|-----------|----------|--------|
| Inverted-list iterators | **C++** | Inner loop over millions of postings |
| B-tree index I/O | **C++** | Low-level disk format, byte manipulation |
| Snappy / VByte compression | **C++** | Bit-twiddling; existing C libs available |
| Scoring functions (BM25, LM) | **C++** | Called per posting |
| Proximity / window operators | **C++** | Stateful iteration with tight loops |
| Document parsing (TREC, WARC) | **Python** | Text-heavy, not latency-critical |
| Query parsing / tree traversal | **Python** | Clarity matters more than speed |
| Stemming | reuse existing libs | `PyStemmer` wraps Snowball C; Krovetz has C port |
| Evaluation metrics | **Python** | Numpy arithmetic; not on the hot path |
| TupleFlow replacement | **Python** | `multiprocessing` + `concurrent.futures` |
| CLI tools | **Python** | argparse, rich progress bars |

---

## 4. Phase Plan

Each phase is independently shippable and testable against the Java version.

---

### Phase 0 — Infrastructure & Skeleton (2–3 weeks, 1 engineer)

**Goal:** Establish the repo layout, build system, and CI before writing any real logic.

#### Tasks
- [ ] Initialize `pygalago/` Python package with `pyproject.toml` (setuptools + CMake extension)
- [ ] Set up `CMakeLists.txt` for `libgalago` C++ shared library + pybind11 bindings
- [ ] Configure GitHub Actions: build matrix (Linux/macOS), run C++ unit tests (Catch2), run Python tests (pytest)
- [ ] Define the top-level Python API namespaces:
  - `pygalago.index` — index building
  - `pygalago.retrieval` — search
  - `pygalago.eval` — metrics
  - `pygalago.tools` — CLI entry points
- [ ] Port `utility/json/Parameters.java` → Python `pygalago.parameters.Parameters` (dict subclass with JSON load/save)
- [ ] Write a golden-output test harness: run Java Galago, capture results, compare against Python port

**Deliverable:** `pip install -e .` works; CI is green; `pygalago.Parameters` passes tests.  
**Effort:** ~3 person-weeks

---

### Phase 1 — Utility & Data Structures (3–4 weeks, 1–2 engineers)

**Goal:** Port the foundational data structures that everything else depends on.

#### 1a. B-tree reader/writer (C++)
The on-disk index format is a custom B-tree. Port:
- `utility/btree/` → `src/btree/btree.{h,cpp}`
- Key operations: open, seek, next, close
- Write C++ unit tests comparing byte-for-byte output against Java-generated index files

#### 1b. Compression (C++)
- VByte integer encoding (`utility/compression/`) → `src/compression/vbyte.{h,cpp}`
- Snappy integration: link against `libsnappy` (C library) — no reimplementation needed
- Delta encoding for posting lists

#### 1c. Buffer management (C++)
- Port `utility/buffer/` → `src/buffer/`
- `DataStream`, `ReadStream`, `WriteStream` equivalents

#### 1d. Python utilities
- `pygalago.parameters`: Parameters class (already in Phase 0)
- `pygalago.common`: CmpUtil, collection utilities
- Type stubs (`.pyi`) for IDE support

**Effort:** ~4 person-weeks  
**Risk:** Index file format must be binary-compatible with existing Galago indexes or a migration tool is needed.

---

### Phase 2 — Index Reader (C++ + Python bindings) (4–5 weeks, 2 engineers)

**Goal:** Read existing Galago-format index files from Python/C++.

#### 2a. Core index interfaces (C++)
- `Index` abstract class → `include/galago/index.h`
- `KeyValueReader`, `KeyIterator` → disk B-tree wrappers

#### 2b. Index part readers (C++)
Port `core/index/disk/`:
- `InvertedListReader` — term → (docid, tf) posting list
- `PositionalListReader` — term → (docid, positions[]) posting list
- `NamesReader` — docid → document name
- `LengthsReader` — docid → document length
- `AggregateReader` — collection statistics (N, avg_dl, df, cf)

#### 2c. In-memory index (C++)
Port `core/index/mem/` for incremental/real-time indexing:
- `MemoryIndex` that can be searched alongside disk index

#### 2d. Python bindings
Expose via pybind11:
```python
import pygalago
idx = pygalago.index.open("/path/to/index")
postings = idx.postings("information")  # iterator
print(idx.lengths.get(42))             # doc length
```

**Effort:** ~5 person-weeks  
**Risk:** B-tree format details are opaque — requires careful reverse-engineering of Java serialization code.

---

### Phase 3 — Iterator Framework (C++) (5–6 weeks, 2 engineers)

**Goal:** Port the 40+ iterator implementations that are the inner loop of every query evaluation.

This is the most performance-critical and architecturally complex phase.

#### 3a. Base iterator abstractions (C++)
Port `core/retrieval/iterator/`:
- `BaseIterator` → pure virtual C++ class with `currentDocId()`, `moveTo(docid)`, `isDone()`
- `CountIterator` — returns term frequency
- `ExtentIterator` — returns positions/extents
- `ScoreIterator` — returns document score

#### 3b. Logical combinators (C++)
- `ConjunctionIterator` — AND of multiple iterators
- `DisjunctionIterator` — OR (merge) of multiple iterators  
- `SynonymIterator` — treat multiple terms as one

#### 3c. Positional / proximity operators (C++)
- `OrderedWindowIterator` — `#od(terms, window)` operator
- `UnorderedWindowIterator` — `#uw(terms, window)` operator
- `InsideIterator` — restricts to extents inside another

#### 3d. Scoring operators (C++)
- `WeightedSumIterator` — combines child scores with weights
- `MaxScoreIterator` — DAAT with upper-bound pruning
- `BM25ScoringIterator`, `LanguageModelScoringIterator`

#### 3e. ScoringContext (C++)
Port `retrieval/processing/ScoringContext.java` → thread-local state struct passed through iterator chain.

#### 3f. Python bindings + tests
Expose a Python-inspectable iterator for debugging:
```python
it = idx.postings_iterator("information")
while not it.done:
    print(it.doc_id, it.count)
    it.next()
```
Golden test: compare every iterator's output against Java on a small index.

**Effort:** ~6 person-weeks  
**Risk:** Extent/window operators carry subtle state invariants; bugs here produce silently wrong rankings.

---

### Phase 4 — Query Processing (Python + C++) (4–5 weeks, 2 engineers)

**Goal:** Parse queries, build and optimize the operator tree, execute ranked retrieval.

#### 4a. Query node representation (Python)
Port `core/retrieval/query/Node.java` → Python `pygalago.query.Node` dataclass tree.  
Query language: `#combine(term1 term2 #od:2(phrase terms))`

#### 4b. Query parser (Python)
- Recursive descent parser for Galago's structured query language
- Output: `Node` tree
- Use `pyparsing` or hand-written LL parser

#### 4c. Traversals / optimizations (Python)
Port `core/retrieval/traversal/` as a pipeline of `NodeVisitor` classes:
- `StopWordTraversal` — remove stop words
- `PartAssignerTraversal` — bind operators to index parts
- `AnnotateParameters` — attach BM25/LM hyper-parameters to nodes
- `FullDependenceTraversal`, `SDMTraversal` — expand to dependency models

#### 4d. Processing models (C++ with Python wrapper)
Port `core/retrieval/processing/`:
- `RankedDocumentModel` — standard DAAT top-k retrieval → C++
- `MaxScoreDocumentModel` — WAND pruning → C++
- `RankedPassageModel` — passage-level scoring → C++

Python wrapper:
```python
results = retrieval.retrieve("information retrieval", n=1000)
```

#### 4e. LocalRetrieval (Python)
Port `LocalRetrieval.java` as a Python class orchestrating:
1. Parse query text → Node tree
2. Run traversals
3. Build C++ iterator tree via pybind11
4. Run processing model → top-k results
5. Return `Results` / `ScoredDocument` list

**Effort:** ~5 person-weeks

---

### Phase 5 — Document Parsing & Indexing (Python + C++) (4–5 weeks, 2 engineers)

**Goal:** Build new indexes from document collections.

#### 5a. Document model (Python)
Port `core/parse/Document.java`:
```python
@dataclass
class Document:
    name: str
    text: str
    terms: list[str]
    tags: list[Tag]
    metadata: dict[str, str]
```

#### 5b. Format parsers (Python)
Port `core/parse/`:
- `TrecTextParser` — TREC `<DOC>...</DOC>` format
- `TrecWebParser` — WT10g/GOV2 format
- `WARCParser` — web crawl archives
- `WikiParser` — Wikipedia XML dumps
- `JSONParser` — JSON document collections

#### 5c. Tokenizer (C++)
Port `core/tokenize/Tokenizer.java` → fast C++ whitespace+punctuation tokenizer callable from Python.

#### 5d. Stemming (reuse)
- Snowball stemmers: use `PyStemmer` package (wraps the same C Snowball code)
- Krovetz: use `KrovetzStemmer` Python package or compile the existing C port

#### 5e. Index builder (Python + C++)
Replace TupleFlow with Python `concurrent.futures`:
- `IndexBuilder` coordinates: parse → tokenize → stem → sort → write
- C++ writer for the B-tree inverted list (performance-critical)
- Corpus writer for document store

Pipeline:
```
documents → [parallel parse workers] → sorted posting streams → [merge] → B-tree index
```

#### 5f. Parallel TupleFlow replacement
Port the conceptual model of TupleFlow (sort → group → process):
- Use `multiprocessing.Pool` + disk-based merge sort
- Or: use Apache Arrow + Parquet for intermediate sort

**Effort:** ~5 person-weeks  
**Risk:** Correct sorted-merge with multi-pass external sort is complex; consider using `sort` utility or Arrow for the sort step.

---

### Phase 6 — Evaluation Module (Python) (1–2 weeks, 1 engineer)

**Goal:** Replicate `eval/` metrics in Python using numpy.

#### Tasks
- [ ] Port all metrics from `eval/metric/`:
  - MAP, R-Precision, Recall@k, P@k
  - NDCG, NDCG@k
  - MRR (Mean Reciprocal Rank)
  - Binary preference (Bpref)
- [ ] Read/write standard TREC qrels and run files
- [ ] Statistical significance: paired t-test, Wilcoxon (use `scipy.stats`)
- [ ] CLI: `pygalago eval --qrels qrels.txt --results run.txt`

**Effort:** ~1.5 person-weeks  
**Note:** `pytrec_eval` (CFFI wrapper around the original C trec_eval) can cover most of this and is already standard in the community. Evaluate whether to use it directly vs. reimplementing.

---

### Phase 7 — CLI Tools & Contrib (Python) (3–4 weeks, 1–2 engineers)

**Goal:** Replace the 40+ `core/tools/` command-line tools and useful `contrib/` features.

#### 7a. Core tools (Python CLI via `click` or `argparse`)
- `pygalago build-index` — full indexing pipeline
- `pygalago search` — interactive search / batch run
- `pygalago batch-search` — run a query file, output TREC run
- `pygalago dump-index` — human-readable index inspection
- `pygalago eval` — evaluation metrics
- `pygalago make-corpus` — corpus format conversion

#### 7b. Contrib: Learning-to-rank features
Port `contrib/learning/`:
- Feature extraction for LTR (BM25, LM scores, field scores, proximity features)
- Output: SVM-rank / LibSVM format

#### 7c. Contrib: Pseudo-relevance feedback
Port `core/retrieval/prf/` and `contrib/retrieval/`:
- RM3 relevance model expansion
- Rocchio feedback

**Effort:** ~3.5 person-weeks

---

### Phase 8 — Integration, Documentation & Release (3–4 weeks)

- [ ] End-to-end golden tests on standard TREC collections (e.g., Robust04, TREC-8)
- [ ] Performance benchmarks: query latency, index build time vs. Java Galago
- [ ] Memory profiling — ensure no leaks in C++ paths
- [ ] Write migration guide: Java Galago → PyGalago
- [ ] `pip install pygalago` via PyPI with pre-built wheels (manylinux2014 + macOS arm64/x86_64)
- [ ] API documentation (Sphinx + autodoc)

**Effort:** ~3.5 person-weeks

---

## 5. Effort Summary

| Phase | Subsystem | Language | Weeks | Engineers | Person-Weeks |
|-------|-----------|----------|-------|-----------|-------------|
| 0 | Infrastructure & skeleton | Python/CMake | 2–3 | 1 | **3** |
| 1 | Utility & data structures | C++ / Python | 3–4 | 1–2 | **5** |
| 2 | Index reader | C++ + bindings | 4–5 | 2 | **8** |
| 3 | Iterator framework | C++ | 5–6 | 2 | **11** |
| 4 | Query processing | Python + C++ | 4–5 | 2 | **9** |
| 5 | Parsing & indexing | Python + C++ | 4–5 | 2 | **9** |
| 6 | Evaluation metrics | Python | 1–2 | 1 | **2** |
| 7 | CLI tools & contrib | Python | 3–4 | 1–2 | **6** |
| 8 | Integration & release | Both | 3–4 | 2 | **7** |
| **Total** | | | **29–38 weeks** | | **~60 person-weeks** |

**Summary:**
- **Optimistic (2 engineers, focused):** ~18 months calendar time
- **Realistic (1–2 engineers, part-time / research cadence):** 2.5–3 years
- **Accelerated (3 engineers):** ~12–14 months

---

## 6. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Index format incompatibility | High | High | Port B-tree reader first; write compatibility tests against real indexes |
| Iterator state bugs → wrong rankings | Medium | High | Golden tests comparing each operator output against Java on small known-output indexes |
| TupleFlow replacement bottleneck | Medium | Medium | Prototype external merge sort early (Phase 1); fall back to single-threaded initially |
| Stemmer behavior drift | Low | Medium | Use same underlying Snowball C source via PyStemmer |
| pybind11 ABI instability | Low | Low | Pin pybind11 version; use stable `py::object` APIs |
| Scope creep from contrib | Medium | Low | Mark contrib as Phase 7+ and cut if timeline slips |
| C++ memory leaks in iterator chain | Medium | Medium | Valgrind + AddressSanitizer in CI from day one |

---

## 7. Recommended Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| C++ standard | C++17 | `std::optional`, `std::string_view`, structured bindings |
| Build system | CMake 3.21+ | scikit-build-core for Python extension |
| Python bindings | pybind11 | Industry standard; good numpy interop |
| Python package | `pyproject.toml` + scikit-build-core | Modern PEP 517 build |
| CLI | Click | Composable command groups |
| Compression | libsnappy (C) | Already used by Java Galago |
| Stemming | PyStemmer (Snowball C) | Same algorithm, no Java |
| IR Evaluation | pytrec_eval or reimpl | Wraps original C trec_eval |
| Testing | Catch2 (C++) + pytest (Python) | Standard for each ecosystem |
| CI | GitHub Actions | Matrix builds for Linux/macOS |
| Packaging | cibuildwheel | Pre-built binary wheels |

---

## 8. Milestone Checkpoints

| Milestone | Deliverable | Target |
|-----------|-------------|--------|
| M1 | Read an existing Galago index; iterate postings in Python | End of Phase 2 |
| M2 | Execute a single-term query end-to-end, match Java ranking | End of Phase 3 |
| M3 | Run full structured queries (BM25, SDM) matching Java | End of Phase 4 |
| M4 | Build a new index from a TREC collection | End of Phase 5 |
| M5 | Reproduce Robust04 MAP within 0.001 of Java results | End of Phase 8 |

---

## 9. What to Skip / Defer

- **TupleFlow distributed modes** (SLURM, Hadoop): Replace with simple `multiprocessing`; distributed indexing can be added later if needed
- **Geometric/spatial indexing** (`core/index/geometric/`): Niche feature; defer to post-1.0
- **Web UI** for job monitoring (`tupleflow/web/`): Out of scope
- **Anchor text extraction** (`core/links/`): Defer; used only for web-scale experiments
- **Stanford CoreNLP dependency**: Remove entirely; replace with spaCy or NLTK where needed
- **Jetty embedded server**: Remove; Python has Flask/FastAPI if needed later

---

## 10. Suggested First Steps (Next 4 Weeks)

1. Set up the repo skeleton and CI (Phase 0) — 1 week
2. Port `Parameters.java` to Python and write tests — 0.5 weeks
3. Implement B-tree reader in C++ and verify it reads a real Galago index — 2 weeks
4. Write the golden-output test harness comparing Java and C++ B-tree output — 0.5 weeks

These first steps derisk the hardest unknown (index format compatibility) immediately.
