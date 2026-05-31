// BSD License (http://www.galagosearch.org/license)
#include "galago/index/postings_reader.h"
#include "galago/compression/vbyte.h"

#include <stdexcept>
#include <cstring>
#include <algorithm>

namespace galago {

// ── Option flags (match BTreeValueSource.java) ────────────────────────────────
static constexpr int HAS_SKIPS    = 0x01;
static constexpr int HAS_MAXTF    = 0x02;
static constexpr int HAS_INLINING = 0x04;

// ── PostingsIterator construction ─────────────────────────────────────────────

PostingsIterator::PostingsIterator(FileStream file,
                                   int64_t value_start,
                                   int64_t value_len,
                                   const std::string& term)
    : done_(false)
    , docs_stream_(file)
    , counts_stream_(file)
    , skips_stream_(file)
    , skip_positions_stream_(file)
{
    stats_.term = term;

    // `file` is already a sub-stream scoped to [value_start, value_end) in the file.
    // All offsets below are relative to this stream's start (= value_start).
    // We use file.sub_stream(rel_offset, length) to create further sub-slices.

    // Read the header from a bounded slice of the value (max 200 bytes).
    FileStream header = file.sub_stream(0, std::min(value_len, (int64_t)200));

    int32_t options = static_cast<int32_t>(header.read_vbyte_u32());
    bool has_inlining = (options & HAS_INLINING) != 0;
    bool has_skips    = (options & HAS_SKIPS)    != 0;
    bool has_maxtf    = (options & HAS_MAXTF)    != 0;

    if (has_inlining) {
        /* inlineMinimum = */ header.read_vbyte_u32();
    }

    stats_.document_count   = static_cast<int64_t>(header.read_vbyte_u64());
    stats_.collection_count = static_cast<int64_t>(header.read_vbyte_u64());

    if (has_maxtf) {
        stats_.max_tf = static_cast<int64_t>(header.read_vbyte_u64());
    } else {
        stats_.max_tf = INT64_MAX;
    }

    if (has_skips) {
        skip_distance_       = static_cast<int64_t>(header.read_vbyte_u64());
        skip_reset_distance_ = static_cast<int64_t>(header.read_vbyte_u64());
        num_skips_           = static_cast<int64_t>(header.read_vbyte_u64());
    }

    int64_t doc_byte_len   = static_cast<int64_t>(header.read_vbyte_u64());
    int64_t count_byte_len = static_cast<int64_t>(header.read_vbyte_u64());
    int64_t pos_byte_len   = static_cast<int64_t>(header.read_vbyte_u64());

    int64_t skips_byte_len          = 0;
    int64_t skip_positions_byte_len = 0;
    if (has_skips) {
        skips_byte_len          = static_cast<int64_t>(header.read_vbyte_u64());
        skip_positions_byte_len = static_cast<int64_t>(header.read_vbyte_u64());
    }

    // All offsets here are relative to value start (i.e. relative to `file`).
    int64_t doc_rel   = header.position();                 // bytes consumed by header
    int64_t count_rel = doc_rel   + doc_byte_len;
    int64_t pos_rel   = count_rel + count_byte_len;

    docs_stream_start_   = doc_rel;
    counts_stream_start_ = count_rel;

    docs_stream_   = file.sub_stream(doc_rel,   doc_byte_len);
    counts_stream_ = file.sub_stream(count_rel, count_byte_len);

    has_skips_ = has_skips;
    if (has_skips) {
        int64_t skips_rel          = pos_rel + pos_byte_len;
        int64_t skip_positions_rel = skips_rel + skips_byte_len;

        skips_stream_          = file.sub_stream(skips_rel, skips_byte_len);
        skip_positions_stream_ = file.sub_stream(skip_positions_rel, skip_positions_byte_len);

        next_skip_doc_     = static_cast<int64_t>(skips_stream_.read_vbyte_u64());
        docs_byte_floor_   = 0;
        counts_byte_floor_ = 0;
    }

    doc_index_ = 0;
    load();
}

// ── Internal: load the next (docid, count) pair ───────────────────────────────

void PostingsIterator::load() {
    if (doc_index_ >= stats_.document_count) {
        done_ = true;
        current_doc_   = INT64_MAX;
        current_count_ = 0;
        return;
    }
    current_doc_ += static_cast<int64_t>(docs_stream_.read_vbyte_u64());
    current_count_ = static_cast<int32_t>(counts_stream_.read_vbyte_u32());
}

// ── next ──────────────────────────────────────────────────────────────────────

void PostingsIterator::next() {
    if (done_) return;
    ++doc_index_;
    load();
}

// ── skip_to ───────────────────────────────────────────────────────────────────

void PostingsIterator::skip_once() {
    if (next_skip_doc_ == INT64_MAX) return;

    // current_skip_pos is an offset within the skip_positions sub-stream
    // (matches Java: skipPositionsStream.seek(currentSkipPosition) where
    //  currentSkipPosition is relative to the start of that byte array).
    int64_t current_skip_pos = last_skip_position_
                               + static_cast<int64_t>(skips_stream_.read_vbyte_u64());

    if (skips_read_ % skip_reset_distance_ == 0) {
        // Tier-2: seek within skip_positions_stream_ to update byte floors.
        skip_positions_stream_.seek(current_skip_pos);
        docs_byte_floor_   = static_cast<int64_t>(skip_positions_stream_.read_vbyte_u64());
        counts_byte_floor_ = static_cast<int64_t>(skip_positions_stream_.read_vbyte_u64());
    }

    current_doc_ = next_skip_doc_;

    if (skips_read_ + 1 == num_skips_) {
        next_skip_doc_ = INT64_MAX;
    } else {
        next_skip_doc_ += static_cast<int64_t>(skips_stream_.read_vbyte_u64());
    }

    ++skips_read_;
    last_skip_position_ = current_skip_pos;
}

void PostingsIterator::synchronize_skip_positions() {
    while (!done_ && next_skip_doc_ <= current_doc_) {
        int64_t saved = current_doc_;
        skip_once();
        current_doc_ = saved;
    }
}

void PostingsIterator::reposition_main_streams() {
    if ((skips_read_ - 1) % skip_reset_distance_ == 0) {
        docs_stream_.seek(docs_byte_floor_);
        counts_stream_.seek(counts_byte_floor_);
    } else {
        skip_positions_stream_.seek(last_skip_position_);
        int64_t doc_delta   = static_cast<int64_t>(skip_positions_stream_.read_vbyte_u64());
        int64_t count_delta = static_cast<int64_t>(skip_positions_stream_.read_vbyte_u64());
        docs_stream_.seek(docs_byte_floor_ + doc_delta);
        counts_stream_.seek(counts_byte_floor_ + count_delta);
    }
    doc_index_ = static_cast<int64_t>(skip_distance_ * skips_read_) - 1;
}

void PostingsIterator::skip_to(int64_t target) {
    if (done_) return;

    if (has_skips_) {
        synchronize_skip_positions();
        if (target > next_skip_doc_) {
            while (!done_ && skips_read_ < num_skips_ && target > next_skip_doc_) {
                skip_once();
            }
            reposition_main_streams();
        }
    }

    // Linear scan from current position.
    while (!done_ && target > current_doc_) {
        doc_index_ = std::min(doc_index_ + 1, stats_.document_count);
        load();
    }
}

// ── PostingsReader ────────────────────────────────────────────────────────────

PostingsReader::PostingsReader(const std::string& path) : reader_(path) {}

std::optional<PostingsIterator>
PostingsReader::make_iterator(DiskBTreeIterator& it) const {
    if (it.is_done()) return std::nullopt;
    std::string term = it.key_string();
    // We need the raw FileStream from inside the reader to slice sub-streams.
    // Pass it the value region via the iterator's file + value_start/end.
    return PostingsIterator(it.value_stream(),
                            0,                    // value_stream starts at 0 within itself
                            it.value_length(),
                            term);
}

std::optional<PostingsIterator>
PostingsReader::get_postings(const std::string& term) const {
    auto it = reader_.get_iterator(term);
    if (!it) return std::nullopt;
    std::string tkey = it->key_string();
    if (tkey != term) return std::nullopt;
    return PostingsIterator(it->value_stream(),
                            0,
                            it->value_length(),
                            term);
}

std::optional<PostingStats>
PostingsReader::get_stats(const std::string& term) const {
    auto pit = get_postings(term);
    if (!pit) return std::nullopt;
    return pit->stats();
}

// ── PostingsReader::read_positions ────────────────────────────────────────────

std::vector<PositionPosting>
PostingsReader::read_positions(const std::string& term) const {
    auto it_opt = reader_.get_iterator(term);
    if (!it_opt) return {};
    auto& bt = *it_opt;
    if (bt.is_done() || bt.key_string() != term) return {};

    // Parse the full posting value sequentially.
    // Header layout matches PostingsIterator constructor, re-parsed here for
    // direct byte-stream access.
    FileStream file = bt.value_stream();
    int64_t    vlen = bt.value_length();

    if (vlen <= 0) return {};

    FileStream hdr = file.sub_stream(0, std::min(vlen, (int64_t)200));

    int32_t options      = static_cast<int32_t>(hdr.read_vbyte_u32());
    bool    has_inlining = (options & 0x04) != 0;
    bool    has_skips    = (options & 0x01) != 0;
    bool    has_maxtf    = (options & 0x02) != 0;

    if (has_inlining) hdr.read_vbyte_u32();   // inlineMinimum (discard)

    int64_t doc_count  = static_cast<int64_t>(hdr.read_vbyte_u64());
    /* coll_count */    hdr.read_vbyte_u64();  // discard

    if (has_maxtf) hdr.read_vbyte_u64();      // max_tf (discard)
    if (has_skips) {
        hdr.read_vbyte_u64();  // skipDistance
        hdr.read_vbyte_u64();  // skipResetDistance
        hdr.read_vbyte_u64();  // numSkips
    }

    int64_t doc_byte_len   = static_cast<int64_t>(hdr.read_vbyte_u64());
    int64_t count_byte_len = static_cast<int64_t>(hdr.read_vbyte_u64());
    int64_t pos_byte_len   = static_cast<int64_t>(hdr.read_vbyte_u64());

    if (pos_byte_len == 0) return {};   // count-only posting list

    int64_t header_end = hdr.position();  // bytes consumed by header

    if (has_skips) {
        // Skip the skip-lengths fields (not needed here).
        hdr.read_vbyte_u64();  // skipsByteLength
        hdr.read_vbyte_u64();  // skipPositionsByteLength
        header_end = hdr.position();
    }

    int64_t doc_off   = header_end;
    int64_t count_off = doc_off   + doc_byte_len;
    int64_t pos_off   = count_off + count_byte_len;

    FileStream doc_s   = file.sub_stream(doc_off,   doc_byte_len);
    FileStream count_s = file.sub_stream(count_off, count_byte_len);
    FileStream pos_s   = file.sub_stream(pos_off,   pos_byte_len);

    std::vector<PositionPosting> result;
    result.reserve(static_cast<size_t>(doc_count));

    int64_t cur_doc = 0;
    for (int64_t i = 0; i < doc_count; ++i) {
        cur_doc += static_cast<int64_t>(doc_s.read_vbyte_u64());
        int32_t tf = static_cast<int32_t>(count_s.read_vbyte_u32());

        std::vector<int32_t> positions;
        positions.reserve(static_cast<size_t>(tf));
        int32_t prev_pos = 0;
        for (int32_t j = 0; j < tf; ++j) {
            int32_t delta = static_cast<int32_t>(pos_s.read_vbyte_u32());
            prev_pos += delta;
            positions.push_back(prev_pos);
        }
        result.push_back({cur_doc, std::move(positions)});
    }
    return result;
}

// ── PostingsReader::read_positions_for ────────────────────────────────────────

std::vector<PositionPosting>
PostingsReader::read_positions_for(const std::string&          term,
                                   const std::vector<int64_t>& doc_ids) const {
    if (doc_ids.empty()) return {};

    auto it_opt = reader_.get_iterator(term);
    if (!it_opt) return {};
    auto& bt = *it_opt;
    if (bt.is_done() || bt.key_string() != term) return {};

    FileStream file = bt.value_stream();
    int64_t    vlen = bt.value_length();
    if (vlen <= 0) return {};

    FileStream hdr = file.sub_stream(0, std::min(vlen, (int64_t)200));

    int32_t options      = static_cast<int32_t>(hdr.read_vbyte_u32());
    bool    has_inlining = (options & 0x04) != 0;
    bool    has_skips    = (options & 0x01) != 0;
    bool    has_maxtf    = (options & 0x02) != 0;

    if (has_inlining) hdr.read_vbyte_u32();

    int64_t doc_count = static_cast<int64_t>(hdr.read_vbyte_u64());
    hdr.read_vbyte_u64();  // coll_count

    if (has_maxtf) hdr.read_vbyte_u64();
    if (has_skips) {
        hdr.read_vbyte_u64();
        hdr.read_vbyte_u64();
        hdr.read_vbyte_u64();
    }

    int64_t doc_byte_len   = static_cast<int64_t>(hdr.read_vbyte_u64());
    int64_t count_byte_len = static_cast<int64_t>(hdr.read_vbyte_u64());
    int64_t pos_byte_len   = static_cast<int64_t>(hdr.read_vbyte_u64());

    if (pos_byte_len == 0) return {};

    int64_t header_end = hdr.position();
    if (has_skips) {
        hdr.read_vbyte_u64();
        hdr.read_vbyte_u64();
        header_end = hdr.position();
    }

    FileStream doc_s   = file.sub_stream(header_end,                           doc_byte_len);
    FileStream count_s = file.sub_stream(header_end + doc_byte_len,            count_byte_len);
    FileStream pos_s   = file.sub_stream(header_end + doc_byte_len + count_byte_len, pos_byte_len);

    std::vector<PositionPosting> result;
    size_t target_idx = 0;

    int64_t cur_doc = 0;
    for (int64_t i = 0; i < doc_count && target_idx < doc_ids.size(); ++i) {
        cur_doc += static_cast<int64_t>(doc_s.read_vbyte_u64());
        int32_t tf = static_cast<int32_t>(count_s.read_vbyte_u32());

        // Skip targets before cur_doc
        while (target_idx < doc_ids.size() && doc_ids[target_idx] < cur_doc)
            ++target_idx;

        if (target_idx < doc_ids.size() && doc_ids[target_idx] == cur_doc) {
            std::vector<int32_t> positions;
            positions.reserve(static_cast<size_t>(tf));
            int32_t prev_pos = 0;
            for (int32_t j = 0; j < tf; ++j) {
                int32_t delta = static_cast<int32_t>(pos_s.read_vbyte_u32());
                prev_pos += delta;
                positions.push_back(prev_pos);
            }
            result.push_back({cur_doc, std::move(positions)});
            ++target_idx;
        } else {
            // Not a target: skip positions for this document
            for (int32_t j = 0; j < tf; ++j)
                pos_s.read_vbyte_u32();
        }
    }
    return result;
}

} // namespace galago
