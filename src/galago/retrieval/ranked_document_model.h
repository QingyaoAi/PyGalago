#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of RankedDocumentModel.java — DAAT (document-at-a-time) top-k retrieval.
//
// High-level search entry point:
//
//   auto results = search(index_path, terms, n=1000, b=0.75, k=1.2);
//   for (auto& [docid, score] : results) { ... }

#include "galago/retrieval/score_iterator.h"
#include "galago/retrieval/lengths_source.h"
#include "galago/index/disk_index.h"

#include <cstdint>
#include <string>
#include <vector>

namespace galago {

// ── Result type ───────────────────────────────────────────────────────────────

struct ScoredDocument {
    int64_t document = 0;
    double  score    = 0.0;

    bool operator<(const ScoredDocument& o) const { return score < o.score; }
    bool operator>(const ScoredDocument& o) const { return score > o.score; }
};

// ── BM25 parameters ───────────────────────────────────────────────────────────

struct BM25Params {
    double b = 0.75;
    double k = 1.2;
    int    n = 1000;           // number of results requested
    std::string postings_part = "postings.krovetz";
};

// ── Retrieval ─────────────────────────────────────────────────────────────────

// Run DAAT BM25 over an index directory.
// terms: already stemmed/normalised query tokens.
// Returns results sorted by descending score (rank 0 = best).
std::vector<ScoredDocument> bm25_search(const std::string&              index_path,
                                         const std::vector<std::string>& terms,
                                         const BM25Params&               params = {});

// Overload that uses an already-open DiskIndex (avoids reopening).
std::vector<ScoredDocument> bm25_search(DiskIndex&                       index,
                                         LengthsSource&                   lengths,
                                         const std::vector<std::string>&  terms,
                                         const BM25Params&                params = {});

// Core DAAT loop — runs over any ScoreIterator tree.
std::vector<ScoredDocument> daat_top_k(ScoreIterator& root, int n);

// ── Weighted BM25 (Phase 4) ───────────────────────────────────────────────────
// Run DAAT BM25 with per-term weights supplied by the Python query pipeline.
// weighted_terms: list of (term, weight) pairs; weights need not sum to 1
//   (they are passed directly to ScoreCombinationIterator which normalises them).
std::vector<ScoredDocument> bm25_search_weighted(
    DiskIndex&                                            index,
    LengthsSource&                                        lengths,
    const std::vector<std::pair<std::string, double>>&    weighted_terms,
    const BM25Params&                                     params = {});

// ── Dirichlet QL parameters ───────────────────────────────────────────────────

struct QLParams {
    double mu   = 2500.0;
    int    n    = 1000;
    std::string postings_part = "postings.krovetz";
};

// ── QL search ────────────────────────────────────────────────────────────────
// DAAT Dirichlet-smoothed Query Likelihood.
// score(d,q) = Σ_t log((tf_t + μ·p_t) / (dl + μ))
// where p_t = cf_t / C (collection language model).
//
// terms: already stemmed/normalised query tokens (duplicates deduplicated).
// Returns results sorted by descending score.
std::vector<ScoredDocument> ql_search(DiskIndex&                      index,
                                       LengthsSource&                  lengths,
                                       const std::vector<std::string>& terms,
                                       const QLParams&                 params = {});

// Overload that opens the index from path (avoids reopening if called once).
std::vector<ScoredDocument> ql_search(const std::string&              index_path,
                                       const std::vector<std::string>& terms,
                                       const QLParams&                 params = {});

// ── Weighted QL ───────────────────────────────────────────────────────────────
// IDF-weighted Dirichlet QL (unigram component of WSDM-Int).
// score(d,q) = Σ_t w_t · log((tf_t + μ·p_t) / (dl + μ))
// weighted_terms: list of (term, weight) pairs; weights are normalised to sum 1.
std::vector<ScoredDocument> ql_search_weighted(
    DiskIndex&                                         index,
    LengthsSource&                                     lengths,
    const std::vector<std::pair<std::string, double>>& weighted_terms,
    const QLParams&                                    params = {});

} // namespace galago
