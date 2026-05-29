// BSD License (http://www.galagosearch.org/license)
#include "galago/retrieval/disjunction_iterator.h"
#include <numeric>

namespace galago {

DisjunctionIterator::DisjunctionIterator(std::vector<BaseIterator*> children)
    : children_(std::move(children))
{
    for (auto* c : children_) {
        if (!c->has_all_candidates()) driving_.push_back(c);
    }
    has_all_candidates_ = driving_.empty();
    if (has_all_candidates_) {
        driving_ = children_;
    }
}

int64_t DisjunctionIterator::current_candidate() const {
    int64_t mn = DONE;
    for (auto* d : driving_) {
        if (!d->is_done()) mn = std::min(mn, d->current_candidate());
    }
    return mn;
}

bool DisjunctionIterator::is_done() const {
    for (auto* d : driving_) if (!d->is_done()) return false;
    return true;
}

bool DisjunctionIterator::has_match(const ScoringContext& ctx) const {
    for (auto* d : driving_) if (d->has_match(ctx)) return true;
    return false;
}

int64_t DisjunctionIterator::total_entries() const {
    int64_t total = 0;
    for (auto* c : children_) {
        if (c->has_all_candidates()) return c->total_entries();
        total += c->total_entries();
    }
    return total;
}

void DisjunctionIterator::sync_to(int64_t doc) {
    for (auto* c : children_) c->sync_to(doc);
}

void DisjunctionIterator::move_past(int64_t doc) {
    for (auto* d : driving_) d->move_past(doc);
}

void DisjunctionIterator::reset() {
    for (auto* c : children_) c->reset();
}

} // namespace galago
