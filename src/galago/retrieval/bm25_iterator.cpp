// BSD License (http://www.galagosearch.org/license)
#include "galago/retrieval/bm25_iterator.h"
#include <cmath>
#include <algorithm>

namespace galago {

BM25Iterator::BM25Iterator(std::unique_ptr<CountIterator> count_iter,
                             const LengthsSource*           lengths,
                             const Params&                  p)
    : count_(std::move(count_iter))
    , lengths_(lengths)
    , b_(p.b)
    , k_(p.k)
    , avg_dl_(p.avg_doc_length)
{
    // Robertson IDF: log((N - df + 0.5) / (df + 0.5))
    // Matches Java BM25ScoringIterator.java exactly (Math.log((N-df+0.5)/(df+0.5))).
    // Previous formula log(N/(df+0.5)) over-weighted medium-frequency terms.
    idf_ = std::log((static_cast<double>(p.doc_count) - static_cast<double>(p.df) + 0.5)
                    / (static_cast<double>(p.df) + 0.5));

    // Score bounds — used by MAXSCORE (Phase 4+)
    // max score: tf = max_tf, dl = 1 (shortest possible document)
    if (p.max_tf > 0) {
        max_score_ = compute_score(static_cast<int32_t>(p.max_tf), 1);
    } else {
        max_score_ = std::numeric_limits<double>::max();
    }
    // min score: tf=0, which yields 0 regardless of idf (background score)
    min_score_ = 0.0;
}

double BM25Iterator::compute_score(int32_t tf, int32_t dl) const {
    double numerator   = static_cast<double>(tf) * (k_ + 1.0);
    double denominator = static_cast<double>(tf)
                       + k_ * (1.0 - b_ + b_ * static_cast<double>(dl) / avg_dl_);
    return idf_ * numerator / denominator;
}

double BM25Iterator::score(const ScoringContext& ctx) {
    int32_t tf = count_->count(ctx);
    if (tf == 0) return min_score_;
    int32_t dl = lengths_->length(ctx.document);
    return compute_score(tf, dl);
}

} // namespace galago
