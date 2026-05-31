# Robust04 SDM/WSDM-Int — Porter2 positional index (titles)

**Index:** `postings.porter` (Porter2, positional)  **Queries:** 249  **μ:** 2500

**SDM weights:** uni=0.85, od=0.1, uw=0.05


## Results vs Paper (Huston & Croft 2014, Table 7)

| Model     | MAP    | NDCG@20 | P@20   | Paper MAP |
|-----------|--------|---------|--------|-----------|
| QL        | 0.2359 | 0.3906  | 0.3384 | 0.252     |
| BM25      | 0.2278 | 0.3864  | 0.3321 | 0.254     |
| SDM       | 0.2496 | 0.4133  | 0.3604 | 0.263     |
| WSDM-Uni  | 0.2288 | 0.3833  | 0.3303 | —     |
| WSDM-Int  | 0.2505 | 0.4115  | 0.3566 | 0.269     |

## Timing

| Model     | Total (s) | Per query (ms) |
|-----------|-----------|----------------|
| QL        |    75.1 |            302 |
| BM25      |    73.6 |            295 |
| SDM       |   122.7 |            493 |
| WSDM-Uni  |    75.3 |            302 |
| WSDM-Int  |   123.2 |            495 |
