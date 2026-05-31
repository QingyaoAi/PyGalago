# Robust04 Retrieval Experiment — PyGalago (descs)

**Collection:** Robust04  (528,155 documents, 252,013,235 tokens)

**Topics:** 249 TREC descs queries

**Index part:** `postings.porter` (Porter2 stemming, INQUERY stops)


## Results vs Paper (Table 7, Huston & Croft 2014)

| Model | Ours MAP | Ours NDCG@20 | Ours P@20 | Paper MAP | Paper NDCG@20 | Paper P@20 |
|-------|--------|--------|--------|--------|--------|--------|
| BM25     | 0.2357 | 0.3933 | 0.3297 | 0.237 | 0.390 | 0.331 |
| QL       | 0.2418 | 0.3902 | 0.3329 | 0.244 | 0.389 | 0.334 |
| SDM      | 0.2418 | 0.3902 | 0.3329 | 0.258 | 0.406 | 0.349 |
| WSDM-Int | 0.2478 | 0.3910 | 0.3307 | 0.278 | 0.428 | 0.365 |
| RM3      | 0.2428 | 0.3902 | 0.3323 | — | — | — |

## Timing

| Model | Total (s) | Per query (ms) |
|-------|-----------|----------------|
| BM25  |   224.9 |            903 |
| QL    |   353.9 |           1421 |
| SDM   |   384.8 |           1545 |
| WSDM  |   362.5 |           1456 |
| RM3   |  1595.4 |           6407 |
