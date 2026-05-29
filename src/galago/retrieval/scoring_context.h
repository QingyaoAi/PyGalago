#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of ScoringContext.java
//
// Shared mutable state passed by reference through the entire iterator tree
// during DAAT evaluation. Extremely hot — kept deliberately minimal.

#include <cstdint>

namespace galago {

struct ScoringContext {
    int64_t document = 0;
    bool    cachable  = true;
};

} // namespace galago
