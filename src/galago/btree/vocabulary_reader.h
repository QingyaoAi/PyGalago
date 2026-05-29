#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of VocabularyReader.java
//
// The vocabulary is a compact block directory: it stores the first key of each
// data block plus the block's file offset and header length.
// Binary search in the vocabulary locates the block that may contain a key.

#include "galago/io/file_stream.h"

#include <cstdint>
#include <string>
#include <vector>

namespace galago {

struct IndexBlockInfo {
    int      slot_id      = 0;
    std::vector<uint8_t> first_key;
    std::vector<uint8_t> next_slot_key;
    int64_t  begin        = 0;   // absolute file offset of block start
    int64_t  length       = 0;   // total byte length of the block
    int32_t  header_length = 0;  // byte length of the key header inside the block
};

class VocabularyReader {
public:
    // Parse the vocabulary section.
    // stream  : positioned at the start of the vocabulary section
    // value_data_end : absolute file offset where data blocks end
    //                  (= vocabulary section start = used to compute last block length)
    VocabularyReader(FileStream stream, int64_t value_data_end);

    // Number of blocks.
    size_t size() const { return slots_.size(); }

    // Return block info for block index id (null-like: returns nullptr if out of range).
    const IndexBlockInfo* get_slot(int id) const;

    // Binary search: find the block whose range contains key.
    // Returns nullptr if vocabulary is empty.
    const IndexBlockInfo* get(const std::vector<uint8_t>& key) const;

    // Binary search starting from min_block.
    const IndexBlockInfo* get(const std::vector<uint8_t>& key, int min_block) const;

    const std::vector<IndexBlockInfo>& slots() const { return slots_; }

private:
    std::vector<IndexBlockInfo> slots_;

    void read(FileStream& stream, int64_t value_data_end);

    static int compare_keys(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b);
};

} // namespace galago
