// BSD License (http://www.galagosearch.org/license)
#include "galago/index/index_writer.h"
#include "galago/compression/vbyte.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <sstream>
#include <stdexcept>

namespace galago {

// ── Option flags (match BTreeValueIterator.java) ──────────────────────────────
static constexpr int HAS_SKIPS    = 0x01;
static constexpr int HAS_MAXTF    = 0x02;
static constexpr int HAS_INLINING = 0x04;

// ── Posting list encoder ──────────────────────────────────────────────────────
// Writes in PositionIndexCountSource format (positionsByteLen=0, no skips):
//
//   VByte(options = HAS_MAXTF)
//   VByte(documentCount)
//   VByte(collectionCount)
//   VByte(maximumCount)
//   VByte(documentByteLength)
//   VByte(countsByteLength)
//   VByte(positionsByteLength = 0)
//   [delta-coded docids]
//   [VByte counts]

std::vector<uint8_t> encode_postings(
        const std::vector<std::pair<int64_t, int32_t>>& postings) {

    if (postings.empty()) return {};

    // Compute stats
    int64_t doc_count  = static_cast<int64_t>(postings.size());
    int64_t coll_count = 0;
    int64_t max_tf     = 0;
    for (auto& [d, c] : postings) {
        coll_count += c;
        if (c > max_tf) max_tf = c;
    }

    // Encode delta-docids and counts into separate byte buffers
    std::vector<uint8_t> doc_bytes, count_bytes;
    int64_t prev_doc = 0;
    for (auto& [d, c] : postings) {
        vbyte_encode_u64(doc_bytes,   static_cast<uint64_t>(d - prev_doc));
        vbyte_encode_u32(count_bytes, static_cast<uint32_t>(c));
        prev_doc = d;
    }

    // Build full posting list value
    std::vector<uint8_t> out;
    vbyte_encode_u32(out, static_cast<uint32_t>(HAS_MAXTF));   // options
    vbyte_encode_u64(out, static_cast<uint64_t>(doc_count));
    vbyte_encode_u64(out, static_cast<uint64_t>(coll_count));
    vbyte_encode_u64(out, static_cast<uint64_t>(max_tf));
    vbyte_encode_u64(out, static_cast<uint64_t>(doc_bytes.size()));
    vbyte_encode_u64(out, static_cast<uint64_t>(count_bytes.size()));
    vbyte_encode_u64(out, 0ULL);   // positionsByteLength = 0
    out.insert(out.end(), doc_bytes.begin(),   doc_bytes.end());
    out.insert(out.end(), count_bytes.begin(), count_bytes.end());
    return out;
}

// ── Big-endian helpers ────────────────────────────────────────────────────────

static void append_be_i32(std::vector<uint8_t>& v, int32_t x) {
    uint32_t u = static_cast<uint32_t>(x);
    v.push_back(static_cast<uint8_t>(u >> 24));
    v.push_back(static_cast<uint8_t>(u >> 16));
    v.push_back(static_cast<uint8_t>(u >>  8));
    v.push_back(static_cast<uint8_t>(u));
}

static void append_be_i64(std::vector<uint8_t>& v, int64_t x) {
    append_be_i32(v, static_cast<int32_t>(static_cast<uint64_t>(x) >> 32));
    append_be_i32(v, static_cast<int32_t>(static_cast<uint64_t>(x) & 0xFFFFFFFFULL));
}

static void append_be_double(std::vector<uint8_t>& v, double d) {
    uint64_t bits;
    std::memcpy(&bits, &d, 8);
    append_be_i64(v, static_cast<int64_t>(bits));
}

// ── Names writer ──────────────────────────────────────────────────────────────

void write_names(const std::string& path,
                 const std::vector<std::string>& names,
                 int64_t first_docid) {
    DiskBTreeWriter writer(path);

    for (size_t i = 0; i < names.size(); ++i) {
        int64_t docid = first_docid + static_cast<int64_t>(i);

        // Key: big-endian int64 docid (matches Utility.fromLong())
        uint64_t uid = static_cast<uint64_t>(docid);
        std::vector<uint8_t> key = {
            static_cast<uint8_t>(uid >> 56),
            static_cast<uint8_t>(uid >> 48),
            static_cast<uint8_t>(uid >> 40),
            static_cast<uint8_t>(uid >> 32),
            static_cast<uint8_t>(uid >> 24),
            static_cast<uint8_t>(uid >> 16),
            static_cast<uint8_t>(uid >>  8),
            static_cast<uint8_t>(uid)
        };

        // Value: UTF-8 name bytes
        std::vector<uint8_t> val(names[i].begin(), names[i].end());

        writer.add(key, val);
    }

    // Manifest matching DiskNameWriter
    std::string manifest = R"({"writerClass":"DiskNameWriter","readerClass":"DiskNameReader"})";
    writer.close(manifest);
}

// ── Lengths writer ────────────────────────────────────────────────────────────
// Value layout matches DiskLengthsWriter.LengthsList.write():
//   8×int64/double (= 64 bytes header)  then  int32[] array

void write_lengths(const std::string& path,
                   const std::vector<int32_t>& lengths,
                   int64_t first_docid,
                   const std::string& field) {

    DiskBTreeWriter writer(path);

    int64_t n             = static_cast<int64_t>(lengths.size());
    int64_t total_docs    = n;
    int64_t non_zero      = 0;
    int64_t coll_len      = 0;
    int64_t max_len       = 0;
    int64_t min_len       = INT64_MAX;

    for (int32_t l : lengths) {
        if (l > 0) {
            ++non_zero;
            coll_len += l;
            if (l > max_len) max_len = l;
            if (l < min_len) min_len = l;
        }
    }
    if (min_len == INT64_MAX) min_len = 0;

    double avg_len = (non_zero > 0) ? static_cast<double>(coll_len) / non_zero : 0.0;

    int64_t last_docid = first_docid + n - 1;

    // Build value bytes
    std::vector<uint8_t> val;
    append_be_i64(val, total_docs);
    append_be_i64(val, non_zero);
    append_be_i64(val, coll_len);
    append_be_double(val, avg_len);
    append_be_i64(val, max_len);
    append_be_i64(val, min_len);
    append_be_i64(val, first_docid);
    append_be_i64(val, last_docid);

    for (int32_t l : lengths) {
        append_be_i32(val, l);
    }

    // Key: field name ("document")
    std::vector<uint8_t> key(field.begin(), field.end());
    writer.add(key, val);

    std::string manifest = R"({"writerClass":"DiskLengthsWriter","readerClass":"DiskLengthsReader"})";
    writer.close(manifest);
}

// ── Positional posting list encoder ──────────────────────────────────────────
// Writes in PositionIndexCountSource format with positionsByteLen > 0:
//
//   VByte(options = HAS_MAXTF)
//   VByte(documentCount)
//   VByte(collectionCount)
//   VByte(maximumCount)
//   VByte(documentByteLength)
//   VByte(countsByteLength)
//   VByte(positionsByteLength)       ← > 0 here (enables read_positions)
//   [delta-coded docids]
//   [VByte counts]
//   [VByte delta-coded positions, reset per document]

std::vector<uint8_t> encode_positional_postings(
        const std::vector<std::pair<int64_t, std::vector<int32_t>>>& postings) {

    if (postings.empty()) return {};

    int64_t doc_count  = static_cast<int64_t>(postings.size());
    int64_t coll_count = 0;
    int64_t max_tf     = 0;
    for (auto& [d, ps] : postings) {
        coll_count += static_cast<int64_t>(ps.size());
        if (static_cast<int64_t>(ps.size()) > max_tf)
            max_tf = static_cast<int64_t>(ps.size());
    }

    std::vector<uint8_t> doc_bytes, count_bytes, pos_bytes;
    int64_t prev_doc = 0;
    for (auto& [d, ps] : postings) {
        vbyte_encode_u64(doc_bytes,   static_cast<uint64_t>(d - prev_doc));
        prev_doc = d;
        vbyte_encode_u32(count_bytes, static_cast<uint32_t>(ps.size()));
        int32_t prev_pos = 0;
        for (int32_t p : ps) {
            vbyte_encode_u32(pos_bytes, static_cast<uint32_t>(p - prev_pos));
            prev_pos = p;
        }
    }

    std::vector<uint8_t> out;
    vbyte_encode_u32(out, static_cast<uint32_t>(HAS_MAXTF));
    vbyte_encode_u64(out, static_cast<uint64_t>(doc_count));
    vbyte_encode_u64(out, static_cast<uint64_t>(coll_count));
    vbyte_encode_u64(out, static_cast<uint64_t>(max_tf));
    vbyte_encode_u64(out, static_cast<uint64_t>(doc_bytes.size()));
    vbyte_encode_u64(out, static_cast<uint64_t>(count_bytes.size()));
    vbyte_encode_u64(out, static_cast<uint64_t>(pos_bytes.size()));
    out.insert(out.end(), doc_bytes.begin(),   doc_bytes.end());
    out.insert(out.end(), count_bytes.begin(), count_bytes.end());
    out.insert(out.end(), pos_bytes.begin(),   pos_bytes.end());
    return out;
}

// ── Postings index writer ─────────────────────────────────────────────────────

void write_postings_index(const std::string& path,
                          const std::vector<TermPostings>& sorted_terms,
                          int64_t total_docs,
                          int64_t collection_length) {
    DiskBTreeWriter writer(path);

    int64_t vocab_count     = 0;
    int64_t highest_df      = 0;
    int64_t highest_freq    = 0;

    for (const auto& tp : sorted_terms) {
        auto val = encode_postings(tp.postings);
        if (val.empty()) continue;

        writer.add(tp.term, val);
        ++vocab_count;

        int64_t df  = static_cast<int64_t>(tp.postings.size());
        int64_t cf  = 0;
        for (auto& [d, c] : tp.postings) cf += c;

        if (df > highest_df)   highest_df   = df;
        if (cf > highest_freq) highest_freq = cf;
    }

    // Build manifest matching what PositionIndexReader / CountIndexReader expects
    std::ostringstream ms;
    ms << "{"
       << "\"writerClass\":\"PositionIndexWriter\","
       << "\"readerClass\":\"PositionIndexReader\","
       << "\"defaultOperator\":\"counts\","
       << "\"statistics/collectionLength\":" << collection_length << ","
       << "\"statistics/vocabCount\":"       << vocab_count       << ","
       << "\"statistics/highestDocumentCount\":" << highest_df    << ","
       << "\"statistics/highestFrequency\":"  << highest_freq     << ","
       << "\"documentCount\":"               << total_docs
       << "}";

    writer.close(ms.str());
}

} // namespace galago
