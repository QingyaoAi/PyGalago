#pragma once
// Port of CountIterator.java

#include "galago/retrieval/base_iterator.h"

namespace galago {

class CountIterator : public BaseIterator {
public:
    // Number of occurrences of this iterator's term in ctx.document.
    virtual int32_t count(const ScoringContext& ctx) const = 0;
};

} // namespace galago
