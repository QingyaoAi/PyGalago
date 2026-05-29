# PyGalago Query Language Reference

PyGalago uses the **Galago Structured Query Language (GSQL)** — a parenthesized prefix notation where every operator begins with `#`. The query parser is a recursive descent LL(1) parser that produces a `Node` tree consumed by the retrieval pipeline.

This document is the authoritative reference for AI agents and developers writing or generating queries.

---

## 1. Grammar

```
query        ::= argument+
argument     ::= restricted
restricted   ::= unrestricted ( "." field_list )*
unrestricted ::= operator | term
operator     ::= "#" name params? "(" argument* ")"
params       ::= ( ":" key ( "=" value )? )+
field_list   ::= ident ( "," ident )*
term         ::= '"' chars '"' | chars
key, value   ::= printable chars  (no space, ":", "=", "(", ")")
name         ::= printable chars  (no space, ":", "(", ")")
```

### Key rules

- **Operators** start with `#` and always have a parenthesized argument list, even if empty: `#combine()`.
- **Parameters** follow the operator name and precede the `(`: `#combine:0=0.8:1=0.2(t1 t2)`.
- **Bare terms** are leaf nodes equivalent to `#text(term)`.
- **Multiple top-level terms** are wrapped implicitly in `#combine`:
  `information retrieval` → `#combine(information retrieval)`.
- **Quoted terms** may contain spaces: `"new york"` is one token.

---

## 2. Operator Reference

### 2.1 `#combine` — Uniform or weighted OR-combination

```
#combine( arg1 arg2 … argN )
#combine:0=w0:1=w1:…:N-1=wN-1( arg1 arg2 … argN )
```

Scores each document as a weighted sum of child scores. Weights are specified by positional index (0-based). If no weights are given, children are weighted uniformly (1/N each). Weights need not sum to 1 — they are normalised internally.

**Examples:**

```
#combine(information retrieval)
#combine:0=0.7:1=0.3(information retrieval)
#combine:0=0.5:1=0.3:2=0.2(information retrieval systems)
```

**Node representation:**

```python
from pygalago.query.parser import parse
node = parse("#combine:0=0.8:1=0.2(information retrieval)")
# node.operator  → "combine"
# node.params    → {"0": 0.8, "1": 0.2}
# node.children  → [Node("text", {"default": "information"}),
#                   Node("text", {"default": "retrieval"})]
```

---

### 2.2 `#weight` — Explicit weight combination (alias for `#combine`)

```
#weight:0=w0:1=w1( arg1 arg2 )
```

Identical to `#combine` with explicit weights. Provided for Galago compatibility.

---

### 2.3 `#od` / `#ordered` — Ordered proximity window

```
#od:N( term1 term2 … termK )
#ordered:N( term1 term2 … termK )
```

Matches documents where all terms appear **in order** within a window of N tokens. The window size N is the positional parameter (`:N` or `:default=N`).

**Example:**

```
#od:1(new york)         → "new" immediately followed by "york" (window=1)
#od:3(information retrieval)  → within 3 tokens, in order
```

> **Implementation note:** Proximity operators require a positional index (not yet written by the current index builder, which produces count-only indexes). When encountered, the retrieval engine drops the proximity node and re-normalises scores with a warning. To use proximity operators, build the index with positional data.

---

### 2.4 `#uw` / `#unordered` — Unordered proximity window

```
#uw:N( term1 term2 … termK )
#unordered:N( term1 term2 … termK )
```

Matches documents where all terms appear **in any order** within a window of N tokens. Conventionally `N = 4 × number_of_terms` for SDM.

**Example:**

```
#uw:8(information retrieval)   → both terms within 8 tokens, any order
```

> Same implementation note as `#od` above.

---

### 2.5 `#fdm` / `#fulldep` — Full Dependence Model (SDM)

```
#fdm( term1 term2 … termK )
#fdm:uniw=0.8:odw=0.15:uww=0.05( term1 term2 … termK )
#fulldep( term1 term2 … termK )
```

Implements Metzler & Croft's Sequential Dependence Model. Expands into a weighted combination of:

1. **Unigram component** — `#combine(t1 t2 … tK)` with weight `uniw` (default 0.80)
2. **Ordered bigrams** — `#combine(#od:1(t1 t2) #od:1(t2 t3) …)` with weight `odw` (default 0.15)
3. **Unordered bigrams** — `#combine(#uw:4n(t1 t2) #uw:4n(t2 t3) …)` with weight `uww` (default 0.05)

**Example:**

```
#fdm(information retrieval systems)
```

Expands to:

```
#combine:0=0.8:1=0.15:2=0.05(
    #combine(information retrieval systems)
    #combine(
        #ordered:1(information retrieval)
        #ordered:1(retrieval systems)
        #ordered:1(information retrieval systems)
    )
    #combine(
        #unordered:8(information retrieval)
        #unordered:8(retrieval systems)
        #unordered:12(information retrieval systems)
    )
)
```

**Custom weights:**

```
#fdm:uniw=0.9:odw=0.07:uww=0.03(information retrieval)
```

**`windowLimit` parameter** — cap the maximum n-gram size:

```
#fdm:windowLimit=2(information retrieval systems)
# Only bigrams are generated; trigrams are skipped.
```

---

### 2.6 `#text` — Explicit leaf term node

```
#text(term)
#text:part=postings.krovetz(term)
```

The base leaf operator. A bare term like `information` is parsed as `#text(information)`. Rarely written explicitly in queries, but appears in the Node tree after parsing.

You can override which index part the term looks up:

```
#text:part=postings(information)          → unstemmed postings
#text:part=postings.krovetz(information)  → Krovetz-stemmed postings
#text:part=postings.porter(information)   → Porter-stemmed postings
```

---

### 2.7 `#counts` — Explicit count iterator node

```
#counts(term)
#counts:part=postings(term)
```

Same as `#text`; requests only term counts (no position data). Used internally after the `PartAssignerTraversal` stamps a `part` parameter.

---

### 2.8 `#extents` — Extent (position) iterator node

```
#extents(term)
```

Requests full positional data for the term. Used by proximity operators. Not yet directly scoreable in the current retrieval engine.

---

### 2.9 Field restriction — `.field` syntax

```
term.field
term.(field1,field2)
```

Restricts a term to a named field (e.g., `title`, `anchor`). Produces an `inside` node wrapping the term.

```
information.title
information.(title,headline)
```

> **Note:** Field indexes are not yet built by the current `IndexBuilder`. This syntax parses correctly but will produce no results unless a field-specific index part exists.

---

## 3. Parameter Syntax

Parameters attach to an operator using `:key=value` immediately after the operator name and before `(`:

```
#combine:0=0.8:1=0.2(t1 t2)
         ^^^^^^^^^^^^^
         params block
```

**Rules:**

- Keys and values may not contain spaces, `:`, `=`, or `(`.
- A parameter without `=value` is treated as a boolean `true`: `#combine:verbose(t1 t2)`.
- The positional parameter for window operators is the `default` key: `#od:1(t1 t2)` → `params["default"] = 1`.
- Numeric values are stored as Python `int` or `float`; string values remain strings.

**Reserved parameter names (set by traversals, not user-supplied):**

| Key | Set by | Meaning |
|---|---|---|
| `part` | `PartAssignerTraversal` | Index part to use for this leaf |
| `collectionLength` | `AnnotateStatsTraversal` | Total tokens in collection |
| `documentCount` | `AnnotateStatsTraversal` | Total documents (N) |
| `nodeDocumentCount` | `AnnotateStatsTraversal` | df for this term |
| `nodeFrequency` | `AnnotateStatsTraversal` | cf for this term |
| `maximumCount` | `AnnotateStatsTraversal` | max tf for this term |
| `b` | `AnnotateStatsTraversal` | BM25 b parameter |
| `k` | `AnnotateStatsTraversal` | BM25 k parameter |

---

## 4. BM25 Scoring Formula

Every leaf node produces a per-document BM25 score. The formula implemented in `src/galago/retrieval/bm25_iterator.h`:

```
idf   = log( N / (df + 0.5) )
tf'   = tf × (k + 1) / ( tf + k × (1 - b + b × dl / avgdl) )
score = idf × tf'
```

Where:
- `N` = total document count (from `LengthStats.total_document_count`)
- `df` = document frequency of the term (from posting list stats)
- `tf` = raw term frequency in the current document (from posting list)
- `dl` = document length in tokens (from `lengths` part)
- `avgdl` = average document length (from `LengthStats.avg_length`)
- `b` = length normalisation parameter (default **0.75**)
- `k` = term saturation parameter (default **1.2**)

**Score combination:** The final document score is the weighted sum of per-term BM25 scores, with weights derived from the `#combine` tree structure and normalised to sum to 1.

---

## 5. Query Processing Pipeline

When `Retrieval.search(query)` is called, the query goes through these stages:

```
query string
    │
    ▼ parse()
Node tree  (raw)
    │
    ▼ PartAssignerTraversal
Node tree  (each leaf has "part" param)
    │
    ▼ AnnotateStatsTraversal
Node tree  (each leaf has BM25 stats)
    │
    ▼ FullDependenceTraversal  (only if #fdm node present)
Node tree  (expanded SDM)
    │
    ▼ node_to_weighted_terms()
[(term, weight), ...]  (de-duplicated, normalised)
    │
    ▼ bm25_search_weighted()  (C++ DAAT)
[ScoredDocument, ...]  (top-k, descending score)
    │
    ▼ name resolution
[(doc_name, score), ...]
```

---

## 6. Python API — Direct Query Tree Manipulation

```python
from pygalago.query.parser import parse, find_query_terms
from pygalago.query.node   import Node

# Parse a query string
node = parse("information retrieval")
print(node)            # #combine(information retrieval)
print(node.operator)   # "combine"
print(len(node.children))  # 2

# Build a tree programmatically
root = Node("combine", {"0": 0.8, "1": 0.2}, [
    Node.text("information"),
    Node.text("retrieval"),
])
print(root)  # #combine:0=0.8:1=0.2(information retrieval)

# Extract all terms
terms = find_query_terms(parse("#fdm(information retrieval systems)"))
# → {"information", "retrieval", "systems"}

# Pretty-print the tree
node = parse("#combine(information #od:1(new york))")
print(node.to_pretty_string())
# #combine
#   information
#   #ordered
#     new
#     york
```

---

## 7. Complete Operator Summary Table

| Operator | Aliases | Description | Status |
|---|---|---|---|
| `#combine` | `#weight` | Weighted sum of child scores | ✓ Implemented |
| `#od` | `#ordered` | Ordered proximity window | ⚠ Parsed, proximity dropped |
| `#uw` | `#unordered` | Unordered proximity window | ⚠ Parsed, proximity dropped |
| `#fdm` | `#fulldep` | Full Dependence Model (SDM) | ✓ Unigram component active |
| `#text` | — | Leaf term (count index) | ✓ Implemented |
| `#counts` | — | Leaf term (count index, explicit) | ✓ Implemented |
| `#extents` | — | Leaf term (positional index) | ⚠ Parsed, not scored |
| `#inside` | — | Restrict to field extents | ⚠ Parsed, not scored |
| `#smoothinside` | — | Smoothed field restriction | ⚠ Parsed, not scored |

> ⚠ = syntactically valid but currently dropped by `node_to_weighted_terms()` because the positional index is not yet built.

---

## 8. Examples for Common Retrieval Tasks

```python
from pygalago.retrieval import Retrieval
r = Retrieval("/path/to/index")

# 1. Simple keyword query (implicit #combine)
r.search("information retrieval")

# 2. Explicit equal-weight combination
r.search("#combine(information retrieval)")

# 3. Weighted combination (favour the first term)
r.search("#combine:0=0.7:1=0.3(information retrieval)")

# 4. SDM / Full Dependence Model expansion
r.search("#fdm(information retrieval systems)")

# 5. Custom SDM weights (more emphasis on unigrams)
r.search("#fdm:uniw=0.9:odw=0.07:uww=0.03(new york crime)")

# 6. Ordered phrase (currently treated as unigrams)
r.search("#od:1(united states)")

# 7. Mixed: phrase + free terms
r.search("#combine(#od:1(new york) crime)")

# 8. Nested combine with explicit BM25 parameters
r.search("#combine:0=0.6:1=0.4(information retrieval)", b=0.5, k=1.5)
```

---

## 9. Serialisation

Nodes serialise back to valid query strings via `str(node)`:

```python
from pygalago.query.parser import parse
q = "#combine:0=0.8:1=0.2(#od:1(new york) retrieval)"
node = parse(q)
assert str(node) == q   # round-trips losslessly
```

Quoted terms are emitted when the term string contains spaces or special characters.
