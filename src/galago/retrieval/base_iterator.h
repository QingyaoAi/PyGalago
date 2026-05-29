#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of BaseIterator.java
//
// Document-ordered iteration contract shared by all query operators.
// All docids are 0-based internal identifiers; INT64_MAX signals "done".

#include "galago/retrieval/scoring_context.h"
#include <cstdint>
#include <limits>

namespace galago {

class BaseIterator {
public:
    virtual ~BaseIterator() = default;

    // Current candidate document id. Returns INT64_MAX when done.
    virtual int64_t current_candidate() const = 0;

    // True when there are no more candidates.
    virtual bool is_done() const = 0;

    // Advance past identifier so the next call to current_candidate() returns
    // a document strictly greater than identifier.
    virtual void move_past(int64_t identifier) = 0;

    // Synchronise ALL child iterators to identifier (even those with
    // has_all_candidates()==true). Used before scoring a candidate.
    virtual void sync_to(int64_t identifier) = 0;

    // True if the iterator has a real (non-background) match at ctx.document.
    virtual bool has_match(const ScoringContext& ctx) const = 0;

    // True if this iterator supplies data for every possible document
    // (e.g. lengths, priors). Such iterators should not drive iteration.
    virtual bool has_all_candidates() const = 0;

    // Over-estimate of the total number of entries.
    virtual int64_t total_entries() const = 0;

    // Restore the iterator to the beginning of its sequence.
    virtual void reset() = 0;

protected:
    static constexpr int64_t DONE = std::numeric_limits<int64_t>::max();
};

} // namespace galago
