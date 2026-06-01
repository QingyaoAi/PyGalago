## Corpus
* /Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-trec

## System to compare
pyserini: https://github.com/castorini/pyserini
pyterrier: https://github.com/terrier-org/pyterrier

## Things to compare
* Corpus indexing speed
* Query runtime
* performance of term-based methods like QL, BM25, SDM

## System Comparison: PyGalago vs Pyserini vs PyTerrier

**Collection:** Robust04 (528,155 documents)  
**Queries:** 249 TREC title queries  
**Stemmer:** Porter2  **Stopping:** INQUERY (PyGalago), Terrier default (PyTerrier)


### Indexing Time

| System     | Time (s) |
|------------|----------|
| PyGalago   | — (pre-built) |
| Pyserini   | 64.2 |
| PyTerrier  | 203.1 |

### Retrieval Effectiveness

| System | Model | MAP | NDCG@20 | P@20 |
|--------|-------|-----|---------|------|
| PyGalago | BM25 | 0.2352 | 0.4001 | 0.3406 |
| PyGalago | QL | 0.2449 | 0.4078 | 0.3488 |
| PyGalago | SDM | 0.2375 | 0.3965 | 0.3416 |
| Pyserini | BM25 | 0.2420 | 0.4130 | 0.3520 |
| Pyserini | QL | 0.2389 | 0.4038 | 0.3434 |
| PyTerrier | BM25 | 0.2356 | 0.4040 | 0.3468 |
| PyTerrier | QL | 0.2313 | 0.3915 | 0.3327 |
| PyTerrier | SDM | 0.2354 | 0.3983 | 0.3384 |

### Query Runtime (ms/query)

| System | Model | ms/query |
|--------|-------|----------|
| PyGalago | BM25 | 104 |
| PyGalago | QL | 148 |
| PyGalago | SDM | 304 |
| Pyserini | BM25 | 13 |
| Pyserini | QL | 13 |
| PyTerrier | BM25 | 17 |
| PyTerrier | QL | 16 |
| PyTerrier | SDM | 28 |
