// BSD License (http://www.galagosearch.org/license)
#include "galago/btree/vocabulary_reader.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace galago {

// ── Key comparison (lexicographic, unsigned bytes) ─────────────────────────────

int VocabularyReader::compare_keys(const std::vector<uint8_t>& a,
                                    const std::vector<uint8_t>& b) {
    size_t min_len = std::min(a.size(), b.size());
    int cmp = std::memcmp(a.data(), b.data(), min_len);
    if (cmp != 0) return cmp;
    if (a.size() < b.size()) return -1;
    if (a.size() > b.size()) return  1;
    return 0;
}

// ── Construction ───────────────────────────────────────────────────────────────

VocabularyReader::VocabularyReader(FileStream stream, int64_t value_data_end) {
    read(stream, value_data_end);
}

void VocabularyReader::read(FileStream& stream, int64_t value_data_end) {
    // Format (matches VocabularyWriter.java + DiskBTreeWriter.close()):
    //   int32  final_key_length
    //   bytes  final_key[final_key_length]
    //   repeat until end of stream:
    //     vbyte  key_length
    //     bytes  key[key_length]
    //     vbyte  block_offset (absolute file position)
    //     vbyte  header_length

    int32_t final_key_length = stream.read_int();
    std::vector<uint8_t> final_key;
    stream.read_fully(final_key, static_cast<size_t>(final_key_length));

    int64_t last_begin = 0;

    while (!stream.is_done()) {
        uint32_t key_len = stream.read_vbyte_u32();
        std::vector<uint8_t> key;
        stream.read_fully(key, key_len);
        int64_t offset        = static_cast<int64_t>(stream.read_vbyte_u64());
        int32_t header_length = static_cast<int32_t>(stream.read_vbyte_u32());

        IndexBlockInfo slot;
        slot.slot_id       = static_cast<int>(slots_.size());
        slot.first_key     = std::move(key);
        slot.begin         = offset;
        slot.header_length = header_length;

        // Set length of previous slot now that we know this slot's start.
        if (!slots_.empty()) {
            slots_.back().length        = offset - last_begin;
            slots_.back().next_slot_key = slot.first_key;
        }

        last_begin = offset;
        slots_.push_back(std::move(slot));
    }

    if (!slots_.empty()) {
        slots_.back().length        = value_data_end - last_begin;
        slots_.back().next_slot_key = final_key;
    }
}

// ── Lookup ─────────────────────────────────────────────────────────────────────

const IndexBlockInfo* VocabularyReader::get_slot(int id) const {
    if (id < 0 || id >= static_cast<int>(slots_.size())) return nullptr;
    return &slots_[static_cast<size_t>(id)];
}

const IndexBlockInfo* VocabularyReader::get(const std::vector<uint8_t>& key) const {
    return get(key, 0);
}

const IndexBlockInfo* VocabularyReader::get(const std::vector<uint8_t>& key, int min_block) const {
    if (slots_.empty()) return nullptr;

    int big   = static_cast<int>(slots_.size()) - 1;
    int small = min_block;

    while (big - small > 1) {
        int mid = small + (big - small) / 2;
        if (compare_keys(slots_[static_cast<size_t>(mid)].first_key, key) <= 0) {
            small = mid;
        } else {
            big = mid;
        }
    }

    const IndexBlockInfo* one = &slots_[static_cast<size_t>(small)];
    const IndexBlockInfo* two = &slots_[static_cast<size_t>(big)];

    return (compare_keys(two->first_key, key) <= 0) ? two : one;
}

} // namespace galago
