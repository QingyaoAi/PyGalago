# Robust04 Retrieval Experiment — PyGalago

**Collection:** Robust04  (528,155 documents, 252,013,235 tokens)

**Topics:** 249 TREC title queries (301–700)

**Qrels:** 311,410 judgments across 249 topics

**Index part:** `postings.krovetz` (Krovetz stemming)


## Model Settings

| Model | Scoring | Parameters |
|-------|---------|------------|
| BM25  | Okapi BM25 | b=0.75, k=1.2 |
| QL    | Dirichlet Query Likelihood | μ=2500 |
| SDM   | Sequential Dependence Model | μ=2500, uni=0.85, od=0.10†, uw=0.05† |
| WSDM  | IDF-Weighted Dirichlet QL | μ=2500, w_t∝log(N/df_t) |
| RM3   | QL + Pseudo-Relevance Feedback | μ=2500, fbDocs=10, fbTerms=20, λ=0.6 |

> † SDM ordered/unordered window features require a positional index. This build contains count-only posting lists, so the bigram components are unavailable — SDM reduces to its unigram (QL) component and is **numerically identical to QL** in this setting.

## Retrieval Results

| Model | MAP | NDCG@10 | NDCG@20 | P@10 | MRR | Bpref | Avg query time |
|-------|--------|--------|--------|--------|--------|--------|----------------|
| BM25  | 0.1721 | 0.3010 | 0.2940 | 0.3088 | 0.4910 | 0.1744 | 127 ms |
| QL    | 0.1814 | 0.3172 | 0.3066 | 0.3193 | 0.5572 | 0.1808 | 207 ms |
| SDM   | 0.1814 | 0.3172 | 0.3066 | 0.3193 | 0.5572 | 0.1808 | 0 ms |
| WSDM  | 0.1743 | 0.3047 | 0.2953 | 0.3040 | 0.5342 | 0.1749 | 207 ms |
| RM3   | 0.1816 | 0.3173 | 0.3068 | 0.3193 | 0.5572 | 0.1808 | 2858 ms |

## Per-Topic Breakdown (MAP)

First 20 topics shown (sorted by topic id).

| Topic | Query | BM25 | QL | SDM | WSDM | RM3 |
|-------|-------|-------|-------|-------|-------|-------|
| 301 | international organized crime | 0.0391 | 0.0511 | 0.0511 | 0.0429 | 0.0508 |
| 302 | poliomyelitis and post polio | 0.4589 | 0.5364 | 0.5364 | 0.5335 | 0.5381 |
| 303 | hubble telescope achievements | 0.1577 | 0.1816 | 0.1816 | 0.1822 | 0.1780 |
| 304 | endangered species mammals | 0.0085 | 0.0069 | 0.0069 | 0.0069 | 0.0069 |
| 305 | most dangerous vehicles | 0.0001 | 0.0001 | 0.0001 | 0.0000 | 0.0001 |
| 306 | african civilian deaths | 0.0135 | 0.0072 | 0.0072 | 0.0072 | 0.0072 |
| 307 | new hydroelectric projects | 0.1806 | 0.1992 | 0.1992 | 0.2093 | 0.1998 |
| 308 | implant dentistry | 0.3617 | 0.4443 | 0.4443 | 0.4257 | 0.4482 |
| 309 | rap and crime | 0.0004 | 0.0008 | 0.0008 | 0.0010 | 0.0008 |
| 310 | radio waves and brain cancer | 0.1579 | 0.0961 | 0.0961 | 0.1015 | 0.0925 |
| 311 | industrial espionage | 0.4555 | 0.5202 | 0.5202 | 0.3409 | 0.5202 |
| 312 | hydroponics | 0.1909 | 0.1909 | 0.1909 | 0.1909 | 0.1909 |
| 313 | magnetic levitation maglev | 0.3198 | 0.3430 | 0.3430 | 0.3452 | 0.3432 |
| 314 | marine vegetation | 0.0007 | 0.0002 | 0.0002 | 0.0001 | 0.0002 |
| 315 | unexplained highway accidents | 0.0013 | 0.0001 | 0.0001 | 0.0001 | 0.0001 |
| 316 | polygamy polyandry polygyny | 0.6144 | 0.5698 | 0.5698 | 0.5698 | 0.5698 |
| 317 | unsolicited faxes | 0.2440 | 0.2225 | 0.2225 | 0.2225 | 0.2225 |
| 318 | best retirement country | 0.0114 | 0.0178 | 0.0178 | 0.0076 | 0.0178 |
| 319 | new fuel sources | 0.0260 | 0.0342 | 0.0342 | 0.0236 | 0.0341 |
| 320 | undersea fiber optic cable | 0.0367 | 0.0462 | 0.0462 | 0.0522 | 0.0464 |

## Notes

- **QL** uses Dirichlet smoothing with μ=2500 (the standard Galago default).
- **WSDM** weights each term by its BM25-style IDF before Dirichlet scoring. This is the unigram component of the Weighted Sequential Dependence Model.
- **RM3** estimates a relevance model from the top-10 QL results, using 500 randomly-sampled content terms as the expansion vocabulary. The expanded query is interpolated with the original query model at λ=0.6.
- **SDM** is reported separately for completeness but is numerically identical to QL in this run because the Robust04 index was built without positional posting data. A full SDM implementation requires reindexing with position lists.
- All models use the **Krovetz-stemmed** (`postings.krovetz`) index part.
- All models retrieve top-1000 documents.
- Evaluation uses the standard Robust04 qrels (311,410 judgments).

## Timing Summary

| Model | Total (s) | Per query (ms) |
|-------|-----------|----------------|
| BM25  |    31.6 |            127 |
| QL    |    51.5 |            207 |
| SDM   |     0.0 |              0 |
| WSDM  |    51.6 |            207 |
| RM3   |   711.8 |           2858 |
