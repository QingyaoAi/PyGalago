// BSD License (http://www.galagosearch.org/license)
#include "galago/index/names_reader.h"

namespace galago {

// ── Key encoding: big-endian int64 (matches Utility.fromLong()) ──────────────

std::vector<uint8_t> NamesReader::docid_to_key(int64_t docid) {
    uint64_t v = static_cast<uint64_t>(docid);
    return {
        static_cast<uint8_t>(v >> 56), static_cast<uint8_t>(v >> 48),
        static_cast<uint8_t>(v >> 40), static_cast<uint8_t>(v >> 32),
        static_cast<uint8_t>(v >> 24), static_cast<uint8_t>(v >> 16),
        static_cast<uint8_t>(v >>  8), static_cast<uint8_t>(v)
    };
}

int64_t NamesReader::key_to_docid(const std::vector<uint8_t>& key) {
    if (key.size() != 8) return -1;
    uint64_t v =
        (static_cast<uint64_t>(key[0]) << 56) |
        (static_cast<uint64_t>(key[1]) << 48) |
        (static_cast<uint64_t>(key[2]) << 40) |
        (static_cast<uint64_t>(key[3]) << 32) |
        (static_cast<uint64_t>(key[4]) << 24) |
        (static_cast<uint64_t>(key[5]) << 16) |
        (static_cast<uint64_t>(key[6]) <<  8) |
        (static_cast<uint64_t>(key[7]));
    return static_cast<int64_t>(v);
}

// ── Construction ──────────────────────────────────────────────────────────────

NamesReader::NamesReader(const std::string& path) : reader_(path) {}

// ── Lookup ────────────────────────────────────────────────────────────────────

std::string NamesReader::get_name(int64_t docid) const {
    auto key = docid_to_key(docid);
    auto it = reader_.get_iterator(key);
    if (!it) return {};
    auto val = it->value_bytes();
    return std::string(reinterpret_cast<const char*>(val.data()), val.size());
}

} // namespace galago
