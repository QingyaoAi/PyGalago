# Robust04 Retrieval Experiment — PyGalago (titles)

**Collection:** Robust04  (528,155 documents, 252,013,235 tokens)

**Topics:** 249 TREC titles queries

**Index part:** `postings.porter` (Porter2 stemming, INQUERY stops)


## Results vs Paper (Table 7, Huston & Croft 2014)

| Model | Ours MAP | Ours NDCG@20 | Ours P@20 | Paper MAP | Paper NDCG@20 | Paper P@20 |
|-------|--------|--------|--------|--------|--------|--------|
| BM25     | 0.2356 | 0.3955 | 0.3418 | 0.254 | 0.412 | 0.363 |
| QL       | 0.2449 | 0.4019 | 0.3488 | 0.252 | 0.412 | 0.365 |
| SDM      | 0.2449 | 0.4019 | 0.3488 | 0.263 | 0.423 | 0.375 |
| WSDM-Int | 0.2367 | 0.3925 | 0.3396 | 0.269 | 0.432 | 0.382 |
| RM3      | 0.2452 | 0.4022 | 0.3494 | — | — | — |

## Timing

| Model | Total (s) | Per query (ms) |
|-------|-----------|----------------|
| BM25  |    38.9 |            156 |
| QL    |    58.4 |            235 |
| SDM   |    72.0 |            289 |
| WSDM  |    57.4 |            231 |
| RM3   |   934.3 |           3752 |
