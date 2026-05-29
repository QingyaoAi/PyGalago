#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DisjunctionIterator.java — OR combinator.
//
// current_candidate() = minimum docid among all driving iterators.
// has_match() = true if ANY driving iterator matches.
// Driving iterators are those with has_all_candidates()==false.

#include "galago/retrieval/base_iterator.h"
#include <memory>
#include <vector>

namespace galago {

// DisjunctionIterator is non-owning: callers keep child iterators alive.
// This lets ScoreCombinationIterator own its children via unique_ptr while
// DisjunctionIterator holds raw-pointer views for iteration.
class DisjunctionIterator : public BaseIterator {
public:
    explicit DisjunctionIterator(std::vector<BaseIterator*> children);

    int64_t current_candidate() const override;
    bool    is_done()           const override;
    bool    has_match(const ScoringContext& ctx) const override;
    bool    has_all_candidates() const override { return has_all_candidates_; }
    int64_t total_entries()     const override;

    void sync_to(int64_t doc)  override;
    void move_past(int64_t doc) override;
    void reset() override;

protected:
    std::vector<BaseIterator*> children_;
    std::vector<BaseIterator*> driving_;
    bool has_all_candidates_ = false;
};

} // namespace galago
