#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of ScoreCombinationIterator.java — the #combine operator.
//
// Computes a weighted sum of child ScoreIterators (normalised to sum-to-1
// by default). Drives from the union (OR) of child posting lists.

#include "galago/retrieval/score_iterator.h"
#include "galago/retrieval/disjunction_iterator.h"

#include <vector>

namespace galago {

class ScoreCombinationIterator : public ScoreIterator {
public:
    // weights: if empty, uniform 1/N weights are used.
    // children: ownership transferred; must all be ScoreIterator subclasses.
    ScoreCombinationIterator(std::vector<ScoreIterator*> score_children,
                              std::vector<double>         weights = {});

    // ── ScoreIterator ─────────────────────────────────────────────────────────
    double score(const ScoringContext& ctx)   override;
    double maximum_score() const override;
    double minimum_score() const override;

    // ── BaseIterator — delegated to inner DisjunctionIterator ─────────────────
    int64_t current_candidate() const override { return disj_.current_candidate(); }
    bool    is_done()           const override { return disj_.is_done(); }
    bool    has_match(const ScoringContext& ctx) const override { return disj_.has_match(ctx); }
    bool    has_all_candidates() const override { return disj_.has_all_candidates(); }
    int64_t total_entries()     const override { return disj_.total_entries(); }

    void sync_to(int64_t doc)   override { disj_.sync_to(doc); }
    void move_past(int64_t doc) override { disj_.move_past(doc); }
    void reset()                override { disj_.reset(); }

private:
    std::vector<ScoreIterator*> scorers_;   // non-owning (owned by disj_ children)
    std::vector<double>         weights_;
    DisjunctionIterator         disj_;
};

} // namespace galago
