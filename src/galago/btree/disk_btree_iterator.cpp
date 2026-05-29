// BSD License (http://www.galagosearch.org/license)
#include "galago/btree/disk_btree_iterator.h"
#include "galago/btree/disk_btree_reader.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

namespace galago {

// ── Key comparison ─────────────────────────────────────────────────────────────

int DiskBTreeIterator::compare_keys(const std::vector<uint8_t>& a,
                                     const std::vector<uint8_t>& b) {
    size_t min_len = std::min(a.size(), b.size());
    int cmp = std::memcmp(a.data(), b.data(), min_len);
    if (cmp != 0) return cmp;
    if (a.size() < b.size()) return -1;
    if (a.size() > b.size()) return  1;
    return 0;
}

// ── Construction ───────────────────────────────────────────────────────────────

DiskBTreeIterator::DiskBTreeIterator(const DiskBTreeReader& reader,
                                      const IndexBlockInfo*  block_info)
    : file_(reader.file_)
    , vocabulary_(&(*reader.vocabulary_))  // optional is guaranteed non-empty here
    , file_length_(reader.file_.file_size())
    , block_stream_(reader.file_)          // placeholder; overwritten in load_block_header
{
    load_block_header(block_info);
}

// ── Block loading ──────────────────────────────────────────────────────────────

void DiskBTreeIterator::load_block_header(const IndexBlockInfo* info) {
    block_info_ = info;

    int64_t abs_start = info->begin;

    // Slice the file into just the header section of this block.
    block_stream_ = file_.absolute_sub_stream(abs_start, abs_start + info->header_length);

    end_value_offset_ = abs_start + info->length;
    key_count_        = static_cast<int>(block_stream_.read_long());

    key_cache_.assign(key_count_, {});
    end_value_offset_cache_.assign(key_count_, 0LL);

    start_value_offset_ = abs_start + info->header_length;
    key_index_          = 0;
    done_               = false;
    cache_key_count_    = 0;

    cache_keys();
}

// ── Key caching ────────────────────────────────────────────────────────────────
// Decode a group of CACHE_GROUP_SIZE entries from the header stream.
// Block header layout (matching DiskBTreeWriter.getBlockHeader()):
//   int64  key_count
//   --- for entry 0: ---
//     vbyte  key_length
//     bytes  key[key_length]
//     vbyte  remaining_value_bytes (totalData - cumulative data so far)
//   --- for entry j > 0: ---
//     vbyte  prefix_overlap
//     vbyte  key_length
//     bytes  key[key_length - prefix_overlap]  (only non-shared suffix)
//     vbyte  remaining_value_bytes

void DiskBTreeIterator::cache_keys() {
    for (int i = 0; i < CACHE_GROUP_SIZE; ++i) {
        if (cache_key_count_ >= key_count_) return;

        if (cache_key_count_ == 0) {
            // First entry: full key
            uint32_t key_len = block_stream_.read_vbyte_u32();
            std::vector<uint8_t> key_bytes;
            block_stream_.read_fully(key_bytes, key_len);
            end_value_offset_cache_[0] = static_cast<int64_t>(block_stream_.read_vbyte_u32());
            key_cache_[0]              = std::move(key_bytes);
            ++cache_key_count_;
        } else {
            // Subsequent entry: prefix-compressed
            uint32_t common    = block_stream_.read_vbyte_u32();
            uint32_t key_len   = block_stream_.read_vbyte_u32();
            if (key_len < common) {
                throw std::runtime_error("DiskBTreeIterator: key_len < common prefix");
            }
            std::vector<uint8_t> key_bytes(key_len);
            // Copy shared prefix from previous key
            const auto& prev = key_cache_[static_cast<size_t>(cache_key_count_ - 1)];
            std::copy(prev.begin(), prev.begin() + common, key_bytes.begin());
            // Read the non-shared suffix
            block_stream_.read_fully(key_bytes.data() + common, key_len - common);

            end_value_offset_cache_[cache_key_count_] =
                static_cast<int64_t>(block_stream_.read_vbyte_u32());
            key_cache_[static_cast<size_t>(cache_key_count_)] = std::move(key_bytes);
            ++cache_key_count_;
        }
    }
}

// ── Value offsets ──────────────────────────────────────────────────────────────
// end_value_offset_cache_[i] = (total value bytes) - (cumulative bytes up to entry i)
// So end_value_offset_cache_[i] = bytes remaining after entry i.

int64_t DiskBTreeIterator::value_start() const {
    if (key_index_ == 0) {
        return start_value_offset_;
    }
    return end_value_offset_ - end_value_offset_cache_[static_cast<size_t>(key_index_ - 1)];
}

int64_t DiskBTreeIterator::value_end() const {
    return end_value_offset_ - end_value_offset_cache_[static_cast<size_t>(key_index_)];
}

std::vector<uint8_t> DiskBTreeIterator::value_bytes() const {
    int64_t start  = value_start();
    int64_t length = value_length();
    std::vector<uint8_t> buf;
    FileStream vs = file_.absolute_sub_stream(start, start + length);
    vs.read_fully(buf, static_cast<size_t>(length));
    return buf;
}

FileStream DiskBTreeIterator::value_stream() const {
    int64_t start = value_start();
    return file_.absolute_sub_stream(start, start + value_length());
}

// ── Navigation ─────────────────────────────────────────────────────────────────

bool DiskBTreeIterator::next_index_block() {
    const IndexBlockInfo* next = vocabulary_->get_slot(block_info_->slot_id + 1);
    if (!next) {
        key_index_ = key_count_ - 1;
        done_ = true;
        return false;
    }
    load_block_header(next);
    return true;
}

bool DiskBTreeIterator::next_key() {
    ++key_index_;
    if (key_index_ >= key_count_) {
        return next_index_block();
    }
    while (key_index_ >= cache_key_count_) {
        cache_keys();
    }
    return true;
}

void DiskBTreeIterator::find(const std::vector<uint8_t>& key) {
    // If the key falls outside this block, load the correct block.
    if (compare_keys(block_info_->first_key, key) > 0 ||
        compare_keys(key, block_info_->next_slot_key) >= 0)
    {
        const IndexBlockInfo* blk = vocabulary_->get(key);
        if (!blk) { done_ = true; return; }
        load_block_header(blk);
    }

    // Allow backward scan within the block (restart from entry 0).
    if (!key_cache_[static_cast<size_t>(key_index_)].empty() &&
        compare_keys(key, key_cache_[static_cast<size_t>(key_index_)]) < 0)
    {
        key_index_ = 0;
    }

    while (key_index_ < key_count_) {
        while (key_index_ >= cache_key_count_) cache_keys();
        if (compare_keys(key_cache_[static_cast<size_t>(key_index_)], key) >= 0) return;
        ++key_index_;
    }
    next_key();
}

void DiskBTreeIterator::skip_to(const std::vector<uint8_t>& key) {
    // Only search forward: if key is beyond this block, advance to the right block.
    if (compare_keys(key, block_info_->next_slot_key) >= 0) {
        const IndexBlockInfo* blk = vocabulary_->get(key, block_info_->slot_id);
        if (!blk) { done_ = true; return; }
        load_block_header(blk);
    }

    while (key_index_ < key_count_) {
        while (key_index_ >= cache_key_count_) cache_keys();
        if (compare_keys(key_cache_[static_cast<size_t>(key_index_)], key) >= 0) return;
        ++key_index_;
    }
    next_key();
}

} // namespace galago
