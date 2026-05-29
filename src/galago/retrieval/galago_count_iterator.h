#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskCountIterator.java — wraps a PostingsIterator as a CountIterator.

#include "galago/retrieval/count_iterator.h"
#include "galago/index/postings_reader.h"
#include <string>

namespace galago {

class GalagoCountIterator : public CountIterator {
public:
    // Construct from a ready PostingsIterator.
    // pr is borrowed (must outlive this iterator); used for reset().
    GalagoCountIterator(PostingsReader* pr,
                        std::string term,
                        PostingsIterator pit);

    // ── CountIterator ─────────────────────────────────────────────────────────
    int32_t count(const ScoringContext& ctx) const override;

    // ── BaseIterator ──────────────────────────────────────────────────────────
    int64_t current_candidate() const override;
    bool    is_done()           const override { return pit_.is_done(); }
    bool    has_all_candidates() const override { return false; }
    int64_t total_entries()     const override { return pit_.stats().document_count; }
    bool    has_match(const ScoringContext& ctx) const override;

    void sync_to(int64_t doc) override;
    void move_past(int64_t doc) override { sync_to(doc + 1); }
    void reset() override;

    const std::string& term() const { return term_; }
    const PostingStats& stats() const { return pit_.stats(); }

private:
    PostingsReader* reader_;  // borrowed — for reset()
    std::string     term_;
    PostingsIterator pit_;
};

} // namespace galago
