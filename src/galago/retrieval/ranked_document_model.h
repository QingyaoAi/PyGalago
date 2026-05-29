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

} // namespace galago
