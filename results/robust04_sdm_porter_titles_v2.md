# Robust04 SDM/WSDM-Int — Porter2 positional index (titles)

**Index:** `postings.porter` (Porter2, positional)  **Queries:** 249  **μ:** 2500

**SDM weights:** uni=0.85, od=0.1, uw=0.05


## Results vs Paper (Huston & Croft 2014, Table 7)

| Model     | MAP    | NDCG@20 | P@20   | Paper MAP |
|-----------|--------|---------|--------|-----------|
| QL        | 0.2455 | 0.4037  | 0.3510 | 0.252     |
| BM25      | 0.2359 | 0.3972  | 0.3434 | 0.254     |
| SDM       | 0.2592 | 0.4230  | 0.3723 | 0.263     |
| WSDM-Uni  | 0.2372 | 0.3947  | 0.3412 | —     |
| WSDM-Int  | 0.2591 | 0.4204  | 0.3667 | 0.269     |

## Timing

| Model     | Total (s) | Per query (ms) |
|-----------|-----------|----------------|
| QL        |    78.2 |            314 |
| BM25      |    75.4 |            303 |
| SDM       |   127.9 |            514 |
| WSDM-Uni  |    77.8 |            313 |
| WSDM-Int  |   128.8 |            517 |
