# Robust04 SDM/WSDM-Int — Porter2 positional index (titles)

**Index:** `postings.porter` (Porter2, positional)  **Queries:** 249  **μ:** 2500

**SDM weights:** uni=0.85, od=0.1, uw=0.05


## Results vs Paper (Huston & Croft 2014, Table 7)

| Model     | MAP    | NDCG@20 | P@20   | Paper MAP |
|-----------|--------|---------|--------|-----------|
| QL        | 0.2523 | 0.4128  | 0.3639 | 0.252     |
| BM25      | 0.2537 | 0.4126  | 0.3592 | 0.254     |
| SDM       | 0.2626 | 0.4233  | 0.3729 | 0.263     |
| WSDM-Uni  | 0.2477 | 0.4067  | 0.3508 | —     |
| WSDM-Int  | 0.2636 | 0.4252  | 0.3745 | 0.269     |

## Timing

| Model     | Total (s) | Per query (ms) |
|-----------|-----------|----------------|
| QL        |    76.7 |            308 |
| BM25      |    76.1 |            306 |
| SDM       |   128.3 |            515 |
| WSDM-Uni  |    77.9 |            313 |
| WSDM-Int  |   128.4 |            516 |
