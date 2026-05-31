# Robust04 SDM/WSDM-Int — Porter2 positional index (descs)

**Index:** `postings.porter` (Porter2, positional)  **Queries:** 249  **μ:** 2500

**SDM weights:** uni=0.85, od=0.1, uw=0.05


## Results vs Paper (Huston & Croft 2014, Table 7)

| Model     | MAP    | NDCG@20 | P@20   | Paper MAP |
|-----------|--------|---------|--------|-----------|
| QL        | 0.2418 | 0.3899  | 0.3329 | 0.244     |
| BM25      | 0.2381 | 0.3981  | 0.3371 | 0.237     |
| SDM       | 0.2546 | 0.4080  | 0.3488 | 0.258     |
| WSDM-Uni  | 0.2477 | 0.3914  | 0.3311 | —     |
| WSDM-Int  | 0.2606 | 0.4116  | 0.3498 | 0.278     |

## Timing

| Model     | Total (s) | Per query (ms) |
|-----------|-----------|----------------|
| QL        |   296.1 |           1189 |
| BM25      |   281.7 |           1131 |
| SDM       |   515.7 |           2071 |
| WSDM-Uni  |   305.4 |           1227 |
| WSDM-Int  |   517.5 |           2078 |
