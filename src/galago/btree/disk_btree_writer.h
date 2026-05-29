#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskBTreeWriter.java
//
// Writes sorted (key, value) pairs to a Galago-format B-tree file.
// Keys MUST be supplied in ascending lexicographic order.
//
// File layout:
//   [data blocks — variable length, ~blockSize bytes each]
//   [final key: int32(len) + bytes]
//   [vocabulary entries: VByte(keyLen) + key + VByte(blockOffset) + VByte(headerLen)]
//   [manifest JSON bytes]
//   [footer: vocab_offset int64 + manifest_offset int64 + blockSize int32 + MAGIC int64]

#include "galago/compression/vbyte.h"
#include "galago/btree/btree_format.h"

#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace galago {

class DiskBTreeWriter {
public:
    static constexpr int DEFAULT_BLOCK_SIZE = 16383;

    // Open path for writing.
    explicit DiskBTreeWriter(const std::string& path,
                              int block_size = DEFAULT_BLOCK_SIZE);

    // Add a (key, value) pair.  Keys must be strictly ascending.
    void add(const std::vector<uint8_t>& key, const std::vector<uint8_t>& value);

    // Convenience: string key.
    void add(const std::string& key, const std::vector<uint8_t>& value);

    // Flush pending data and write the footer.  manifest_json is stored
    // verbatim between the vocabulary and the footer.
    void close(const std::string& manifest_json = "{}");

    bool is_open() const { return out_.is_open(); }

private:
    struct PendingEntry {
        std::vector<uint8_t> key;
        std::vector<uint8_t> value;
    };

    // Vocabulary accumulator — in-memory for typical collections.
    struct VocabEntry {
        std::vector<uint8_t> first_key;
        int64_t              offset;
        int32_t              header_length;
    };

    std::ofstream           out_;
    int                     block_size_;
    int64_t                 file_position_ = 0;
    int64_t                 buffered_bytes_ = 0;
    int64_t                 key_count_      = 0;
    int64_t                 block_count_    = 0;

    std::vector<PendingEntry> pending_;
    std::vector<VocabEntry>   vocabulary_;
    std::vector<uint8_t>      prev_key_;
    std::vector<uint8_t>      last_key_;   // last key added (for final-key increment)

    void flush_block();
    std::vector<uint8_t> make_block_header() const;
    int prefix_overlap(const std::vector<uint8_t>& a,
                       const std::vector<uint8_t>& b) const;
    static std::vector<uint8_t> increment_key(const std::vector<uint8_t>& key);
    static void write_be_i32(std::ostream& o, int32_t v);
    static void write_be_i64(std::ostream& o, int64_t v);
    int64_t pending_data_size() const;
    bool needs_flush(const std::vector<uint8_t>& value) const;
};

} // namespace galago
