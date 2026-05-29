#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskNameReader.java
//
// The `names` B-tree stores:
//   key  : docid as 8-byte big-endian int64
//   value: UTF-8 document name string

#include "galago/btree/disk_btree_reader.h"
#include <optional>
#include <string>
#include <cstdint>

namespace galago {

class NamesReader {
public:
    explicit NamesReader(const std::string& path);

    // Return the document name for the given internal docid, or "" if not found.
    std::string get_name(int64_t docid) const;

    // Iterate all (docid, name) pairs in docid order.
    // Calls fn(docid, name) for each entry; stop early by returning false.
    template<typename Fn>
    void for_each(Fn fn) const {
        auto it = reader_.get_iterator();
        if (!it) return;
        while (!it->is_done()) {
            int64_t id = key_to_docid(it->key());
            std::string name(reinterpret_cast<const char*>(it->value_bytes().data()),
                             it->value_length());
            if (!fn(id, name)) break;
            it->next_key();
        }
    }

    const std::string& manifest_json() const { return reader_.manifest_json(); }

private:
    DiskBTreeReader reader_;

    static std::vector<uint8_t> docid_to_key(int64_t docid);
    static int64_t key_to_docid(const std::vector<uint8_t>& key);
};

} // namespace galago
