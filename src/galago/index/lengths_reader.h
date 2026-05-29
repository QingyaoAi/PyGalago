#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskLengthsReader.java + DiskLengthSource.java
//
// The `lengths` B-tree stores:
//   key  : field name ("document") as UTF-8 bytes
//   value: binary blob —
//     8×int64 stats header (big-endian):
//       totalDocumentCount, nonZeroDocumentCount, collectionLength,
//       avgLength (as double bits), maxLength, minLength,
//       firstDocument, lastDocument
//     int32[] lengths  (one per document from firstDocument to lastDocument,
//                       big-endian, stored contiguously)

#include "galago/btree/disk_btree_reader.h"
#include <cstdint>
#include <string>
#include <vector>

namespace galago {

struct LengthStats {
    std::string field_name;
    int64_t total_document_count  = 0;
    int64_t non_zero_doc_count    = 0;
    int64_t collection_length     = 0;
    double  avg_length            = 0.0;
    int64_t max_length            = 0;
    int64_t min_length            = 0;
    int64_t first_document        = 0;
    int64_t last_document         = 0;
};

class LengthsReader {
public:
    // Open a `lengths` B-tree file.
    explicit LengthsReader(const std::string& path);

    // Return the document length for docid, 0 if out of range.
    int32_t get_length(int64_t docid) const;

    // Return stats for a field (default: "document").
    LengthStats get_stats(const std::string& field = "document") const;

    // Total number of documents indexed.
    int64_t total_documents(const std::string& field = "document") const;

    // Iterate all (docid, length) pairs in docid order for a field.
    template<typename Fn>
    void for_each(Fn fn, const std::string& field = "document") const {
        auto it = reader_.get_iterator(field);
        if (!it) return;
        auto val = it->value_bytes();
        if (val.size() < 64) return;

        int64_t first = read_be_i64(val, 56);
        int64_t last  = read_be_i64(val, 64 - 8);  // offset 56 + 8 = 64... no
        // header: 6 int64 + 1 double (8) + 2 int64 = 8 * 8 = 64 bytes
        first = read_be_i64(val, 48);
        last  = read_be_i64(val, 56);
        size_t data_offset = 64;

        for (int64_t d = first; d <= last && data_offset + 4 <= val.size(); ++d) {
            int32_t len = read_be_i32(val, data_offset);
            data_offset += 4;
            if (!fn(d, len)) break;
        }
    }

    const std::string& manifest_json() const { return reader_.manifest_json(); }

private:
    DiskBTreeReader reader_;

    // Helpers to read big-endian values from a byte vector.
    static int64_t read_be_i64(const std::vector<uint8_t>& v, size_t off);
    static int32_t read_be_i32(const std::vector<uint8_t>& v, size_t off);
    static double  read_be_double(const std::vector<uint8_t>& v, size_t off);
};

} // namespace galago
