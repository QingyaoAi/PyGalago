#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of PositionIndexCountSource.java (and CountIndexCountSource.java).
//
// Posting list binary format (inside a B-tree value):
//
//   VByte int32  options  (bitmask: HAS_SKIPS=1, HAS_MAXTF=2, HAS_INLINING=4)
//   [if HAS_INLINING]  VByte int32 inlineMinimum
//   VByte int64  documentCount
//   VByte int64  collectionCount
//   [if HAS_MAXTF]     VByte int64 maximumPositionCount
//   [if HAS_SKIPS]     VByte int64 skipDistance
//                      VByte int64 skipResetDistance
//                      VByte int64 numSkips
//   VByte int64  documentByteLength
//   VByte int64  countsByteLength
//   VByte int64  positionsByteLength    (present even in count-only index)
//   [if HAS_SKIPS]     VByte int64 skipsByteLength
//                      VByte int64 skipPositionsByteLength
//
//   [documentByteLength bytes] delta-VByte doc-ids (delta from previous)
//   [countsByteLength bytes]   VByte term frequencies
//   [positionsByteLength bytes] VByte positions (ignored in count mode)
//   [skip data if HAS_SKIPS]

#include "galago/btree/disk_btree_reader.h"
#include "galago/io/file_stream.h"
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace galago {

// ── Posting stats ─────────────────────────────────────────────────────────────
struct PostingStats {
    std::string term;
    int64_t document_count    = 0;  // df
    int64_t collection_count  = 0;  // cf
    int64_t max_tf            = 0;
};

// ── PostingsIterator ─────────────────────────────────────────────────────────
// Iterates over (docid, count) pairs for a single term.
// Supports skip_to(docid) for DAAT traversal.

class PostingsIterator {
public:
    PostingsIterator() : done_(true) {}

    // Build from a B-tree value stream (the raw posting list bytes).
    // vs: file stream positioned at the start of the value
    // value_len: total byte length of the value
    PostingsIterator(FileStream file,
                     int64_t value_start,
                     int64_t value_len,
                     const std::string& term);

    bool     is_done()    const { return done_; }
    int64_t  doc_id()     const { return current_doc_; }
    int32_t  count()      const { return current_count_; }
    const PostingStats& stats() const { return stats_; }

    // Advance to next posting.
    void next();

    // Advance to the first posting with docid >= target.
    void skip_to(int64_t target);

private:
    bool    done_          = true;
    int64_t current_doc_   = 0;
    int32_t current_count_ = 0;
    int64_t doc_index_     = 0;
    PostingStats stats_;

    // Three parallel byte-streams within the value region of the file.
    FileStream docs_stream_;
    FileStream counts_stream_;

    // Skip list streams (optional)
    bool      has_skips_         = false;
    FileStream skips_stream_;
    FileStream skip_positions_stream_;
    int64_t   skip_distance_       = 0;
    int64_t   skip_reset_distance_ = 0;
    int64_t   num_skips_           = 0;
    int64_t   skips_read_          = 0;
    int64_t   next_skip_doc_       = 0;
    int64_t   last_skip_position_  = 0;
    int64_t   docs_byte_floor_     = 0;
    int64_t   counts_byte_floor_   = 0;
    int64_t   docs_stream_start_   = 0;   // absolute file position
    int64_t   counts_stream_start_ = 0;

    void load();
    void skip_once();
    void synchronize_skip_positions();
    void reposition_main_streams();
};

// ── PositionPosting ───────────────────────────────────────────────────────────
// A (docid, positions) pair for SDM / proximity scoring.
struct PositionPosting {
    int64_t               doc_id;
    std::vector<int32_t>  positions;   // sorted ascending
};

// ── PostingsReader ────────────────────────────────────────────────────────────
// Opens a postings B-tree file and provides per-term iterators.

class PostingsReader {
public:
    explicit PostingsReader(const std::string& path);

    // Return an iterator over postings for term, or nullopt if not in index.
    std::optional<PostingsIterator> get_postings(const std::string& term) const;

    // Stats for a term (df, cf) without full iteration.
    std::optional<PostingStats> get_stats(const std::string& term) const;

    // Read full positional posting list for term (sequential, no skipping).
    // Returns empty vector if the term is not in the index or has no positions.
    std::vector<PositionPosting> read_positions(const std::string& term) const;

    // Read positions only for documents in `doc_ids` (sorted ascending).
    // Much faster than read_positions() for sparse candidate sets.
    std::vector<PositionPosting> read_positions_for(
        const std::string&            term,
        const std::vector<int64_t>&   doc_ids) const;

    const std::string& manifest_json() const { return reader_.manifest_json(); }

private:
    DiskBTreeReader reader_;
    std::optional<PostingsIterator> make_iterator(DiskBTreeIterator& it) const;
};

} // namespace galago
