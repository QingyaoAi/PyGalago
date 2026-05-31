# Robust04 SDM/WSDM-Int — Porter2 positional index (descs)

**Index:** `postings.porter` (Porter2, positional)  **Queries:** 249  **μ:** 2500

**SDM weights:** uni=0.85, od=0.1, uw=0.05


## Results vs Paper (Huston & Croft 2014, Table 7)

| Model     | MAP    | NDCG@20 | P@20   | Paper MAP |
|-----------|--------|---------|--------|-----------|
| QL        | 0.2351 | 0.3774  | 0.3209 | 0.244     |
| BM25      | 0.2262 | 0.3805  | 0.3175 | 0.237     |
| SDM       | 0.2460 | 0.3980  | 0.3422 | 0.258     |
| WSDM-Uni  | 0.2396 | 0.3786  | 0.3207 | —     |
| WSDM-Int  | 0.2521 | 0.3979  | 0.3367 | 0.278     |

## Timing

| Model     | Total (s) | Per query (ms) |
|-----------|-----------|----------------|
| QL        |   295.4 |           1186 |
| BM25      |   276.7 |           1111 |
| SDM       |   510.6 |           2051 |
| WSDM-Uni  |   305.6 |           1227 |
| WSDM-Int  |   511.0 |           2052 |
