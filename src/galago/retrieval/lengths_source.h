#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskLengthSource.java — but loads the entire array into RAM for O(1)
// random access, which is practical for collections up to ~100M documents.
//
// Also satisfies the BaseIterator contract so it can be used in DAAT trees.

#include "galago/retrieval/base_iterator.h"
#include "galago/index/lengths_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace galago {

class LengthsSource : public BaseIterator {
public:
    // Load lengths for the given field from the file at path.
    explicit LengthsSource(const std::string& path,
                           const std::string& field = "document");

    // Construct directly from a LengthsReader (borrows it — reader must outlive this).
    LengthsSource(const LengthsReader& reader,
                  const std::string& field = "document");

    // Random-access length lookup by docid — the primary use in scoring.
    int32_t length(int64_t docid) const;
    int32_t length(const ScoringContext& ctx) const { return length(ctx.document); }

    // ── BaseIterator ─────────────────────────────────────────────────────────
    // LengthsSource is an "all-candidates" iterator: it has data for every
    // document and should never drive DAAT iteration.

    int64_t current_candidate() const override { return current_doc_; }
    bool    is_done()           const override { return current_doc_ > last_doc_; }
    bool    has_all_candidates() const override { return true; }
    int64_t total_entries()     const override { return static_cast<int64_t>(lengths_.size()); }

    bool has_match(const ScoringContext& ctx) const override {
        return ctx.document >= first_doc_ && ctx.document <= last_doc_;
    }

    void sync_to(int64_t doc) override {
        if (doc >= first_doc_ && doc <= last_doc_) current_doc_ = doc;
        else if (doc > last_doc_) current_doc_ = last_doc_ + 1;
    }

    void move_past(int64_t doc) override { sync_to(doc + 1); }

    void reset() override { current_doc_ = first_doc_; }

    const LengthStats& stats() const { return stats_; }

private:
    std::vector<int32_t> lengths_;   // index by (docid - first_doc_)
    int64_t first_doc_ = 0;
    int64_t last_doc_  = 0;
    int64_t current_doc_ = 0;
    LengthStats stats_;

    void load(const LengthsReader& reader, const std::string& field);
};

} // namespace galago
