// BSD License (http://www.galagosearch.org/license)
#include "galago/index/lengths_reader.h"

#include <cstring>
#include <stdexcept>

namespace galago {

// ── Big-endian helpers ────────────────────────────────────────────────────────

int64_t LengthsReader::read_be_i64(const std::vector<uint8_t>& v, size_t off) {
    if (off + 8 > v.size()) throw std::out_of_range("lengths header truncated");
    return static_cast<int64_t>(
        (static_cast<uint64_t>(v[off+0]) << 56) |
        (static_cast<uint64_t>(v[off+1]) << 48) |
        (static_cast<uint64_t>(v[off+2]) << 40) |
        (static_cast<uint64_t>(v[off+3]) << 32) |
        (static_cast<uint64_t>(v[off+4]) << 24) |
        (static_cast<uint64_t>(v[off+5]) << 16) |
        (static_cast<uint64_t>(v[off+6]) <<  8) |
        (static_cast<uint64_t>(v[off+7]))
    );
}

int32_t LengthsReader::read_be_i32(const std::vector<uint8_t>& v, size_t off) {
    if (off + 4 > v.size()) throw std::out_of_range("lengths data truncated");
    return static_cast<int32_t>(
        (static_cast<uint32_t>(v[off+0]) << 24) |
        (static_cast<uint32_t>(v[off+1]) << 16) |
        (static_cast<uint32_t>(v[off+2]) <<  8) |
        (static_cast<uint32_t>(v[off+3]))
    );
}

double LengthsReader::read_be_double(const std::vector<uint8_t>& v, size_t off) {
    uint64_t bits = static_cast<uint64_t>(
        (static_cast<uint64_t>(v[off+0]) << 56) |
        (static_cast<uint64_t>(v[off+1]) << 48) |
        (static_cast<uint64_t>(v[off+2]) << 40) |
        (static_cast<uint64_t>(v[off+3]) << 32) |
        (static_cast<uint64_t>(v[off+4]) << 24) |
        (static_cast<uint64_t>(v[off+5]) << 16) |
        (static_cast<uint64_t>(v[off+6]) <<  8) |
        (static_cast<uint64_t>(v[off+7]))
    );
    double d;
    std::memcpy(&d, &bits, 8);
    return d;
}

// Lengths value header layout (matches DiskLengthSource.reset()):
//   offset  0: int64 totalDocumentCount
//   offset  8: int64 nonZeroDocumentCount
//   offset 16: int64 collectionLength
//   offset 24: double avgLength        (8 bytes, big-endian IEEE 754)
//   offset 32: int64 maxLength
//   offset 40: int64 minLength
//   offset 48: int64 firstDocument
//   offset 56: int64 lastDocument
//   offset 64: int32[] lengths (one per document)

static constexpr size_t HEADER_SIZE = 64;

// ── Construction ──────────────────────────────────────────────────────────────

LengthsReader::LengthsReader(const std::string& path) : reader_(path) {}

// ── get_length ────────────────────────────────────────────────────────────────

int32_t LengthsReader::get_length(int64_t docid) const {
    auto it = reader_.get_iterator(std::string("document"));
    if (!it) return 0;

    auto val = it->value_bytes();
    if (val.size() < HEADER_SIZE) return 0;

    int64_t first = read_be_i64(val, 48);
    int64_t last  = read_be_i64(val, 56);

    if (docid < first || docid > last) return 0;

    size_t array_off = HEADER_SIZE + static_cast<size_t>(docid - first) * 4;
    if (array_off + 4 > val.size()) return 0;

    return read_be_i32(val, array_off);
}

// ── get_stats ─────────────────────────────────────────────────────────────────

LengthStats LengthsReader::get_stats(const std::string& field) const {
    auto it = reader_.get_iterator(field);
    if (!it) return {};

    auto val = it->value_bytes();
    if (val.size() < HEADER_SIZE) return {};

    LengthStats s;
    s.field_name           = field;
    s.total_document_count = read_be_i64(val,  0);
    s.non_zero_doc_count   = read_be_i64(val,  8);
    s.collection_length    = read_be_i64(val, 16);
    s.avg_length           = read_be_double(val, 24);
    s.max_length           = read_be_i64(val, 32);
    s.min_length           = read_be_i64(val, 40);
    s.first_document       = read_be_i64(val, 48);
    s.last_document        = read_be_i64(val, 56);
    return s;
}

// ── total_documents ───────────────────────────────────────────────────────────

int64_t LengthsReader::total_documents(const std::string& field) const {
    return get_stats(field).total_document_count;
}

} // namespace galago
