#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of BM25ScoringIterator.java
//
// Wraps a CountIterator (posting list) and a LengthsSource to produce
// per-document BM25 scores.
//
// BM25 formula (Okapi, Robertson et al.):
//   idf  = log(N / (df + 0.5))
//   tf'  = tf * (k+1) / (tf + k*(1 - b + b*dl/avgdl))
//   score = idf * tf'
//
// Defaults: b=0.75, k=1.2  (matching Galago's BM25ScoringIterator)

#include "galago/retrieval/score_iterator.h"
#include "galago/retrieval/count_iterator.h"
#include "galago/retrieval/lengths_source.h"

#include <cmath>
#include <memory>
#include <string>

namespace galago {

class BM25Iterator : public ScoreIterator {
public:
    struct Params {
        double b = 0.75;
        double k = 1.2;
        // Populated from index statistics:
        double avg_doc_length  = 0.0;
        int64_t doc_count      = 0;
        int64_t df             = 0;   // document frequency
        int64_t max_tf         = 0;   // max term frequency in any document
    };

    BM25Iterator(std::unique_ptr<CountIterator> count_iter,
                 const LengthsSource*           lengths,
                 const Params&                  p);

    // ── ScoreIterator ─────────────────────────────────────────────────────────
    double score(const ScoringContext& ctx) override;
    double maximum_score() const override { return max_score_; }
    double minimum_score() const override { return min_score_; }

    // ── BaseIterator ──────────────────────────────────────────────────────────
    int64_t current_candidate() const override { return count_->current_candidate(); }
    bool    is_done()           const override { return count_->is_done(); }
    bool    has_match(const ScoringContext& ctx) const override { return count_->has_match(ctx); }
    bool    has_all_candidates() const override { return false; }
    int64_t total_entries()     const override { return count_->total_entries(); }

    void sync_to(int64_t doc) override  { count_->sync_to(doc); }
    void move_past(int64_t doc) override { count_->move_past(doc); }
    void reset() override               { count_->reset(); }

    double idf() const { return idf_; }

private:
    std::unique_ptr<CountIterator> count_;
    const LengthsSource* lengths_;  // non-owning

    double b_, k_, avg_dl_, idf_;
    double min_score_, max_score_;

    double compute_score(int32_t tf, int32_t dl) const;
};

} // namespace galago
