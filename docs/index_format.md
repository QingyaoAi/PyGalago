# PyGalago Index Format Reference

This document describes the binary on-disk format of every index part that PyGalago reads and writes. It is aimed at developers who need to inspect, debug, or build tools around the index files, and at AI agents that need to understand what the index contains.

PyGalago's index format is **binary-compatible with Java Galago 3.x**: files written by one can be read by the other without conversion.

---

## 1. Index Directory Layout

An index is a directory containing named files (called *parts*):

```
my-index/
├── buildManifest.json      ← build metadata (JSON)
├── names                   ← docid → document name
├── lengths                 ← field → document length array
├── postings                ← term → posting list (unstemmed)
├── postings.krovetz        ← term → posting list (Krovetz-stemmed)
└── postings.porter         ← term → posting list (Porter-stemmed, optional)
```

Every file except `buildManifest.json` is a **Galago B-tree** (see §2).

### `buildManifest.json`

Written by `IndexBuilder.build()`. Example:

```json
{
  "indexPath": "/path/to/my-index",
  "documentCount": 528155,
  "collectionLength": 252013235,
  "stemmer": "krovetz",
  "buildTime": "2026-05-29T12:00:00Z"
}
```

---

## 2. Galago B-tree File Format

Every index part is a custom B-tree where keys and values are arbitrary byte arrays. Keys are stored in **strictly ascending lexicographic order** across all blocks.

### 2.1 File layout

```
┌──────────────────────────────────┐
│  data blocks (0..N-1)            │ ← variable length
├──────────────────────────────────┤
│  vocabulary section              │
│    int32 BE: final-key length    │
│    final-key bytes               │
│    per-block entries:            │
│      VByte: first-key length     │
│      first-key bytes             │
│      VByte: block file offset    │
│      VByte: block header length  │
├──────────────────────────────────┤
│  manifest JSON bytes             │
├──────────────────────────────────┤
│  footer (28 bytes)               │
│    int64 BE: vocab_offset        │
│    int64 BE: manifest_offset     │
│    int32 BE: block_size          │
│    int64 BE: MAGIC               │
└──────────────────────────────────┘

MAGIC = 0x1a2b3c4d5e6f7a8d
```

The footer is always the **last 28 bytes** of the file. A reader locates the vocabulary and manifest from `vocab_offset` and `manifest_offset` stored in the footer.

### 2.2 Data block layout

Each block stores a contiguous run of sorted (key, value) pairs. The default block size is **16,383 bytes**.

```
┌──────────────────────────────────────────────────────┐
│  HEADER                                              │
│    int64 BE: key_count  (number of entries)          │
│    entry 0:                                          │
│      VByte: full_key_length                          │
│      full_key_length bytes: key0                     │
│      VByte: remaining_value_bytes_after_this_entry   │
│    entry 1..N-1:                                     │
│      VByte: prefix_overlap_with_prev_key             │
│      VByte: full_key_length                          │
│      (key_length - prefix_overlap) bytes: key suffix │
│      VByte: remaining_value_bytes_after_this_entry   │
├──────────────────────────────────────────────────────┤
│  VALUE DATA  (concatenated, no padding)              │
│    value0_bytes                                      │
│    value1_bytes                                      │
│    …                                                 │
└──────────────────────────────────────────────────────┘
```

**Key prefix compression**: consecutive keys that share a common prefix are stored with only the differing suffix, saving space for ordered string keys (e.g., posting list terms).

**`remaining_value_bytes`**: the total byte length of all value data *after* this entry's value. This allows a reader to compute `value_start = header_end + (total_value_bytes - remaining - this_value_len)`.

### 2.3 VByte encoding

Variable-length integer encoding used throughout:

```
value 0–127   → 1 byte:  [0xxxxxxx]
value 128–16383 → 2 bytes: [1xxxxxxx] [0xxxxxxx]  (big-endian 7-bit groups)
…
```

Each byte contributes 7 bits of the integer. The MSB is 1 if more bytes follow, 0 if this is the last byte. The 7-bit groups are big-endian (most significant first).

Implemented in `src/galago/compression/vbyte.h` / `vbyte.cpp`.

---

## 3. `names` Part — docid → document name

**Manifest:**
```json
{"writerClass":"DiskNameWriter","readerClass":"DiskNameReader"}
```

**Key:** Big-endian int64 docid (8 bytes). Docids start at 0.

```
byte[0] = (docid >> 56) & 0xFF
byte[1] = (docid >> 48) & 0xFF
…
byte[7] = (docid)       & 0xFF
```

**Value:** Raw UTF-8 bytes of the document name string (e.g., `FBIS3-1`).

Since docids are stored big-endian, they sort numerically in the B-tree (docid 0 < docid 1 < … < docid N-1). This allows binary search by docid.

**Reading (Python):**
```python
import pygalago._galago as g
nr = g.NamesReader("/path/to/index/names")
name = nr.get_name(0)       # → "FT931-1"
```

**Writing (Python):**
```python
g.write_names("/path/to/index/names", ["doc1", "doc2", "doc3"])
```

---

## 4. `lengths` Part — field → document length array

**Manifest:**
```json
{"writerClass":"DiskLengthsWriter","readerClass":"DiskLengthsReader"}
```

**Key:** Field name as UTF-8 bytes (e.g., `document` for the whole-document length).

**Value layout:**
```
offset  type          description
  0     int64 BE      total_document_count
  8     int64 BE      non_zero_doc_count
  16    int64 BE      collection_length  (sum of all lengths)
  24    double BE     avg_length         (collection_length / non_zero_doc_count)
  32    int64 BE      max_length
  40    int64 BE      min_length
  48    int64 BE      first_docid
  56    int64 BE      last_docid
  64    int32 BE[N]   lengths[0], lengths[1], …, lengths[N-1]
```

The header is exactly 64 bytes (8 values × 8 bytes). The remaining bytes are a packed `int32[]` array of document lengths, indexed by `(docid - first_docid)`.

**Reading (Python):**
```python
lr = g.LengthsReader("/path/to/index/lengths")
length = lr.get_length(42)           # → 913
stats  = lr.get_stats("document")    # → LengthStats object
print(stats.collection_length)       # → 252013235
print(stats.avg_length)              # → 477.158
```

---

## 5. `postings.*` Parts — term → posting list

**Manifest:**
```json
{
  "writerClass":"PositionIndexWriter",
  "readerClass":"PositionIndexReader",
  "defaultOperator":"counts",
  "statistics/collectionLength": 252013235,
  "statistics/vocabCount": 831468,
  "statistics/highestDocumentCount": 495082,
  "statistics/highestFrequency": 5482766,
  "documentCount": 528155
}
```

**Key:** Term string as UTF-8 bytes (the stemmed or unstemmed term text).

**Value (posting list) layout:**
```
VByte  options              (bitmask: HAS_MAXTF=0x02)
VByte  document_count       (df — number of documents containing this term)
VByte  collection_count     (cf — total occurrences across all documents)
VByte  maximum_count        (max_tf — highest tf in any single document)
VByte  document_byte_length (byte length of the delta-docid section)
VByte  count_byte_length    (byte length of the term-count section)
VByte  position_byte_length (0 for count-only indexes; >0 for positional)

[document_byte_length bytes]  delta-coded docids (VByte)
[count_byte_length bytes]     term counts per document (VByte)
[position_byte_length bytes]  positions (not yet written by IndexBuilder)
```

**Delta coding for docids:**
Docids are stored as deltas from the previous docid (the first delta is from 0). For a posting list `[(0, 3), (5, 1), (12, 2)]`:
```
delta-docids:  VByte(0), VByte(5), VByte(7)   ← 5-0=5, 12-5=7
term-counts:   VByte(3), VByte(1), VByte(2)
```

**`options` bitmask:**
| Bit | Value | Meaning |
|-----|-------|---------|
| 0   | 0x01  | HAS_SKIPS — skip list is present (not written by IndexBuilder) |
| 1   | 0x02  | HAS_MAXTF — maximum_count field is present |
| 2   | 0x04  | HAS_INLINING — values are inlined in the vocabulary |

The current `IndexBuilder` always writes `options = HAS_MAXTF (0x02)`.

**Reading (Python):**
```python
pr = g.PostingsReader("/path/to/index/postings.krovetz")

it = pr.get_postings("inform")   # stemmed term
while not it.is_done:
    print(it.doc_id, it.count)   # docid (int), tf (int)
    it.next()

# Skip to a docid ≥ 10000 (used in DAAT retrieval)
it = pr.get_postings("inform")
it.skip_to(10000)
if not it.is_done:
    print(it.doc_id)   # first docid ≥ 10000

# Per-term statistics (without iterating)
stats = pr.get_stats("inform")
# → {"term": "inform", "document_count": 68145, "collection_count": 163441}
```

---

## 6. Internal C++ Types (Python-accessible via pybind11)

### `pygalago._galago.DiskIndex`

Top-level index accessor. Opens all available parts from a directory.

| Method / Property | Description |
|---|---|
| `DiskIndex(path)` | Open an index directory |
| `.get_name(docid)` | `str` — document name for internal docid |
| `.get_length(docid)` | `int` — token count for internal docid |
| `.get_length_stats(field="document")` | `LengthStats` object |
| `.total_documents()` | `int` — total document count |
| `.has_names()` | `bool` |
| `.has_lengths()` | `bool` |
| `.has_postings(part="postings.krovetz")` | `bool` |
| `.get_postings(term, part="postings.krovetz")` | `PostingsIterator` or `None` |
| `.postings_reader(part="postings.krovetz")` | `PostingsReader` or `None` |
| `.path` | `str` — index directory path |

### `pygalago._galago.LengthStats`

| Property | Type | Description |
|---|---|---|
| `.field_name` | `str` | Field name (e.g., `"document"`) |
| `.total_document_count` | `int` | Total documents N |
| `.non_zero_doc_count` | `int` | Documents with length > 0 |
| `.collection_length` | `int` | Sum of all document lengths |
| `.avg_length` | `float` | Average document length |
| `.max_length` | `int` | Maximum document length |
| `.min_length` | `int` | Minimum document length |
| `.first_document` | `int` | First docid with data |
| `.last_document` | `int` | Last docid with data |

### `pygalago._galago.PostingsIterator`

| Method / Property | Description |
|---|---|
| `.is_done` | `bool` — true when exhausted |
| `.doc_id` | `int` — current document id |
| `.count` | `int` — current term frequency |
| `.next()` | Advance to next posting |
| `.skip_to(docid)` | Skip to first docid ≥ target |
| `.stats` | `dict` with `term`, `document_count`, `collection_count`, `max_tf` |
| Iterable | `for doc_id, count in it:` |

### `pygalago._galago.LengthsSource`

Preloads the entire lengths array into RAM for fast random access (used by the retrieval engine). Backed by the `lengths` B-tree file.

```python
ls = g.LengthsSource("/path/to/index/lengths", "document")
print(ls.length(42))   # → 913
print(ls.stats.avg_length)
```

---

## 7. Index Writer Functions (Python-callable)

These are lower-level write functions, useful if you need to build index parts programmatically without using `IndexBuilder`.

```python
import pygalago._galago as g

# Write names part
g.write_names(
    path="/path/to/index/names",
    names=["doc0", "doc1", "doc2"],
    first_docid=0,           # default 0
)

# Write lengths part
g.write_lengths(
    path="/path/to/index/lengths",
    lengths=[913, 250, 432],   # int32 per document
    first_docid=0,
    field="document",          # default "document"
)

# Write postings part
# term_postings: list of (term, [(docid, count), ...]) sorted by term
g.write_postings_index(
    path="/path/to/index/postings",
    term_postings=[
        ("information", [(0, 3), (2, 1)]),
        ("retrieval",   [(0, 2), (1, 1), (2, 1)]),
    ],
    total_docs=3,
    collection_length=18,
)
```

---

## 8. B-tree Reader (Low-Level)

For raw inspection of any index part:

```python
import pygalago._galago as g

reader = g.BTreeReader("/path/to/index/postings.krovetz")

# Print the manifest JSON
print(reader.manifest_json)

# Iterate all key/value pairs
it = reader.iterator()
while not it.is_done:
    print(repr(it.key))           # term string
    print(it.value_length, "bytes")
    it.next_key()

# Seek to a specific key
it = reader.get("information")
if it is not None:
    raw_bytes = it.value           # raw value bytes
```

This is what the CLI `dump-index` command uses internally.

---

## 9. Summary: Part → Reader Mapping

| File | Key type | Value type | Python reader |
|---|---|---|---|
| `names` | `int64 BE docid` | UTF-8 name | `NamesReader` |
| `lengths` | UTF-8 field name | 64-byte header + `int32[]` | `LengthsReader` |
| `postings*` | UTF-8 term | Encoded posting list | `PostingsReader` |
| Any part | `bytes` | `bytes` | `BTreeReader` |
