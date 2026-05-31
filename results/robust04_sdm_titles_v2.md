# Robust04 SDM/WSDM-Int — titles (complete index with positions)

**Index:** `postings.krovetz` (Krovetz, positional)  **Queries:** 249  **μ:** 2500

**SDM weights:** uni=0.85, od=0.1, uw=0.05


## Results vs Paper

| Model | MAP | NDCG@20 | P@20 | Paper MAP |
|-------|-----|---------|------|-----------|
| QL        | 0.2416 | 0.3971 | 0.3442 | 0.252 |
| SDM       | 0.2380 | 0.3932 | 0.3442 | 0.263 |
| WSDM-Uni  | 0.2327 | 0.3818 | 0.3271 | — |
| WSDM-Int  | 0.2288 | 0.3755 | 0.3245 | 0.269 |

## Timing

| Model | Total (s) | Per query (ms) |
|-------|-----------|----------------|
| QL        |    35.3 |            142 |
| SDM       |    66.4 |            267 |
| WSDM-Uni  |    35.8 |            144 |
| WSDM-Int  |    66.3 |            266 |
