// BSD License (http://www.galagosearch.org/license)
#include "galago/retrieval/score_combination_iterator.h"
#include <numeric>
#include <stdexcept>

namespace galago {

ScoreCombinationIterator::ScoreCombinationIterator(
        std::vector<ScoreIterator*> score_children,
        std::vector<double>         weights)
    : scorers_(score_children)
    , disj_(std::vector<BaseIterator*>(score_children.begin(), score_children.end()))
{
    if (scorers_.empty()) throw std::invalid_argument("#combine: no children");

    if (weights.empty()) {
        double w = 1.0 / static_cast<double>(scorers_.size());
        weights_.assign(scorers_.size(), w);
    } else {
        if (weights.size() != scorers_.size())
            throw std::invalid_argument("#combine: weight count != child count");
        double sum = std::accumulate(weights.begin(), weights.end(), 0.0);
        weights_.resize(weights.size());
        for (size_t i = 0; i < weights.size(); ++i)
            weights_[i] = (sum > 0.0) ? weights[i] / sum : weights[i];
    }
}

double ScoreCombinationIterator::score(const ScoringContext& ctx) {
    double total = 0.0;
    for (size_t i = 0; i < scorers_.size(); ++i)
        total += weights_[i] * scorers_[i]->score(ctx);
    return total;
}

double ScoreCombinationIterator::maximum_score() const {
    double mx = 0.0;
    for (size_t i = 0; i < scorers_.size(); ++i)
        mx += weights_[i] * scorers_[i]->maximum_score();
    return mx;
}

double ScoreCombinationIterator::minimum_score() const {
    double mn = 0.0;
    for (size_t i = 0; i < scorers_.size(); ++i)
        mn += weights_[i] * scorers_[i]->minimum_score();
    return mn;
}

} // namespace galago
