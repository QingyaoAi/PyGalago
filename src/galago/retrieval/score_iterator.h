#pragma once
// Port of ScoreIterator.java

#include "galago/retrieval/base_iterator.h"
#include <limits>

namespace galago {

class ScoreIterator : public BaseIterator {
public:
    // Score for ctx.document. Only valid when has_match(ctx) is true;
    // otherwise returns a background (minimum) score.
    virtual double score(const ScoringContext& ctx) = 0;

    // Upper bound on score(); used by MAXSCORE / WAND pruning.
    virtual double maximum_score() const { return std::numeric_limits<double>::max(); }

    // Lower bound on score().
    virtual double minimum_score() const { return std::numeric_limits<double>::lowest(); }
};

} // namespace galago
