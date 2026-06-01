# Robust04 Retrieval Experiment — PyGalago (titles)

**Collection:** Robust04  (528,155 documents, 252,013,235 tokens)

**Topics:** 249 TREC titles queries

**Index part:** `postings.porter` (Porter2 stemming, INQUERY stops)


## Results vs Paper (Table 7, Huston & Croft 2014)

| Model | Ours MAP | Ours NDCG@20 | Ours P@20 | Paper MAP | Paper NDCG@20 | Paper P@20 |
|-------|--------|--------|--------|--------|--------|--------|
| BM25     | 0.2352 | 0.3950 | 0.3406 | 0.254 | 0.412 | 0.363 |
| QL       | 0.2449 | 0.4019 | 0.3488 | 0.252 | 0.412 | 0.365 |
| SDM      | 0.2375 | 0.3907 | 0.3416 | 0.263 | 0.423 | 0.375 |
| WSDM-Int | 0.2358 | 0.3894 | 0.3361 | 0.269 | 0.432 | 0.382 |
| RM3      | 0.2452 | 0.4022 | 0.3494 | — | — | — |

## Timing

| Model | Total (s) | Per query (ms) |
|-------|-----------|----------------|
| BM25  |     3.1 |             13 |
| QL    |     2.3 |              9 |
| SDM   |     8.8 |             35 |
| WSDM  |     2.3 |              9 |
| RM3   |    62.4 |            251 |
