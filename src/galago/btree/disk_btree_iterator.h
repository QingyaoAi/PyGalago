#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskBTreeIterator.java
//
// Iterates over key-value pairs in a disk B-tree in sorted order.
// The value for each key is a byte range [value_start(), value_end()) in the
// underlying file; callers retrieve the bytes with value_bytes().

#include "galago/io/file_stream.h"
#include "galago/btree/vocabulary_reader.h"

#include <cstdint>
#include <string>
#include <vector>

namespace galago {

class DiskBTreeReader;

class DiskBTreeIterator {
public:
    DiskBTreeIterator(const DiskBTreeReader& reader,
                      const IndexBlockInfo*  block_info);

    // ── Navigation ────────────────────────────────────────────────────────────

    // Advance to the next key. Returns false when there are no more keys.
    bool next_key();

    // Seek to the first key >= target (restart from beginning of index).
    void find(const std::vector<uint8_t>& key);

    // Like find() but only searches forward from the current block.
    void skip_to(const std::vector<uint8_t>& key);

    // ── State ─────────────────────────────────────────────────────────────────

    bool is_done() const { return done_; }

    const std::vector<uint8_t>& key() const { return key_cache_[key_index_]; }

    std::string key_string() const {
        const auto& k = key_cache_[key_index_];
        return {reinterpret_cast<const char*>(k.data()), k.size()};
    }

    // Absolute file offsets of value bytes for the current key.
    int64_t value_start() const;
    int64_t value_end()   const;
    int64_t value_length() const { return value_end() - value_start(); }

    // Copy the value bytes into a vector.
    std::vector<uint8_t> value_bytes() const;

    // Return a FileStream sliced to exactly the current value.
    FileStream value_stream() const;

private:
    // Shared reference to the open file (all iterators on the same reader share it).
    FileStream file_;

    // Non-owning pointer into the reader's vocabulary (reader must outlive iterator).
    const VocabularyReader* vocabulary_;
    int64_t file_length_;

    const IndexBlockInfo* block_info_;   // current block
    int64_t start_value_offset_;         // absolute start of value section in block
    int64_t end_value_offset_;           // absolute end of value section (= block end)

    // Per-key metadata decoded from the block header.
    // end_value_offset_cache_[i] = bytes of value data remaining after entry i
    std::vector<int64_t>              end_value_offset_cache_;
    std::vector<std::vector<uint8_t>> key_cache_;

    int key_index_ = 0;
    int key_count_ = 0;
    int cache_key_count_ = 0;
    bool done_ = false;

    FileStream block_stream_;  // stream positioned at the block header

    static constexpr int CACHE_GROUP_SIZE = 5;

    void load_block_header(const IndexBlockInfo* info);
    bool next_index_block();
    void cache_keys();

    static int compare_keys(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);
};

} // namespace galago
