# Robust04 SDM/WSDM-Int — titles (complete index with positions)

**Index:** `postings.krovetz` (Krovetz, positional)  **Queries:** 249  **μ:** 2500

**SDM weights:** uni=0.85, od=0.1, uw=0.05


## Results vs Paper

| Model | MAP | NDCG@20 | P@20 | Paper MAP |
|-------|-----|---------|------|-----------|
| QL        | 0.2202 | 0.3703 | 0.3213 | 0.252 |
| SDM       | 0.2142 | 0.3602 | 0.3157 | 0.263 |
| WSDM-Uni  | 0.2110 | 0.3548 | 0.3038 | — |
| WSDM-Int  | 0.2037 | 0.3435 | 0.2964 | 0.269 |

## Timing

| Model | Total (s) | Per query (ms) |
|-------|-----------|----------------|
| QL        |    48.8 |            196 |
| SDM       |   101.8 |            409 |
| WSDM-Uni  |    51.2 |            206 |
| WSDM-Int  |   101.9 |            409 |
