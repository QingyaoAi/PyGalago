#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskBTreeReader.java

#include "galago/io/file_stream.h"
#include "galago/btree/vocabulary_reader.h"
#include "galago/btree/disk_btree_iterator.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace galago {

class DiskBTreeReader {
public:
    explicit DiskBTreeReader(const std::string& path);

    // True if the magic number at the end of the file matches.
    static bool is_btree(const std::string& path);

    // Return the manifest JSON string.
    const std::string& manifest_json() const { return manifest_json_; }

    // Return an iterator to the first key, or nullopt if the index is empty.
    std::optional<DiskBTreeIterator> get_iterator() const;

    // Return an iterator positioned exactly at key, or nullopt if not found.
    std::optional<DiskBTreeIterator> get_iterator(const std::vector<uint8_t>& key) const;
    std::optional<DiskBTreeIterator> get_iterator(const std::string& key) const;

private:
    friend class DiskBTreeIterator;

    FileStream                       file_;
    std::optional<VocabularyReader>  vocabulary_;
    std::string                      manifest_json_;
    bool                             empty_ = false;

    void init_from_file();
};

} // namespace galago
