// BSD License (http://www.galagosearch.org/license)
#include "galago/btree/disk_btree_reader.h"
#include "galago/btree/btree_format.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>

namespace galago {

// ── Key comparison helper ──────────────────────────────────────────────────────

static int cmp_keys(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b) {
    size_t min_len = std::min(a.size(), b.size());
    int cmp = std::memcmp(a.data(), b.data(), min_len);
    if (cmp != 0) return cmp;
    if (a.size() < b.size()) return -1;
    if (a.size() > b.size()) return  1;
    return 0;
}

// ── Construction ───────────────────────────────────────────────────────────────
// Footer layout (last 28 bytes of file):
//   int64  vocabulary_offset
//   int64  manifest_offset
//   int32  block_size
//   int64  MAGIC

DiskBTreeReader::DiskBTreeReader(const std::string& path)
    : file_(path)
{
    init_from_file();
}

void DiskBTreeReader::init_from_file() {
    int64_t file_len   = file_.file_size();
    int64_t footer_off = file_len - FOOTER_SIZE;

    FileStream footer = file_.absolute_sub_stream(footer_off, file_len);
    int64_t vocab_offset    = footer.read_long();
    int64_t manifest_offset = footer.read_long();
    /* int32_t block_size   = */ footer.read_int();
    int64_t magic           = footer.read_long();

    if (magic != BTREE_MAGIC) {
        throw std::runtime_error(
            "DiskBTreeReader: bad magic — not a Galago B-tree index file");
    }

    // Read manifest JSON
    int64_t manifest_len = footer_off - manifest_offset;
    FileStream mstream = file_.absolute_sub_stream(manifest_offset,
                                                    manifest_offset + manifest_len);
    std::vector<uint8_t> mbytes;
    mstream.read_fully(mbytes, static_cast<size_t>(manifest_len));
    manifest_json_ = std::string(reinterpret_cast<const char*>(mbytes.data()),
                                  mbytes.size());

    // Check emptyIndexFile in manifest (simple string search, no JSON parser)
    auto pos = manifest_json_.find("\"emptyIndexFile\"");
    if (pos != std::string::npos) {
        auto after = manifest_json_.find("true", pos);
        if (after != std::string::npos && after < pos + 40) {
            empty_ = true;
        }
    }

    // Parse vocabulary
    int64_t vocab_len = manifest_offset - vocab_offset;
    FileStream vocab_stream = file_.absolute_sub_stream(vocab_offset,
                                                         vocab_offset + vocab_len);
    // value_data_end = vocab_offset (data blocks end where vocabulary begins)
    vocabulary_.emplace(std::move(vocab_stream), vocab_offset);
}

// ── is_btree ───────────────────────────────────────────────────────────────────

bool DiskBTreeReader::is_btree(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return false;
    std::fseek(f, 0, SEEK_END);
    long len = std::ftell(f);
    if (len < 8) { std::fclose(f); return false; }
    std::fseek(f, len - 8, SEEK_SET);
    uint8_t buf[8];
    size_t got = std::fread(buf, 1, 8, f);
    std::fclose(f);
    if (got != 8) return false;
    int64_t magic = (static_cast<int64_t>(buf[0]) << 56) |
                    (static_cast<int64_t>(buf[1]) << 48) |
                    (static_cast<int64_t>(buf[2]) << 40) |
                    (static_cast<int64_t>(buf[3]) << 32) |
                    (static_cast<int64_t>(buf[4]) << 24) |
                    (static_cast<int64_t>(buf[5]) << 16) |
                    (static_cast<int64_t>(buf[6]) <<  8) |
                    (static_cast<int64_t>(buf[7]));
    return magic == BTREE_MAGIC;
}

// ── Iterators ──────────────────────────────────────────────────────────────────

std::optional<DiskBTreeIterator> DiskBTreeReader::get_iterator() const {
    if (empty_ || !vocabulary_ || vocabulary_->size() == 0) return std::nullopt;
    const IndexBlockInfo* slot = vocabulary_->get_slot(0);
    if (!slot) return std::nullopt;
    return DiskBTreeIterator(*this, slot);
}

std::optional<DiskBTreeIterator>
DiskBTreeReader::get_iterator(const std::vector<uint8_t>& key) const {
    if (empty_ || !vocabulary_) return std::nullopt;
    const IndexBlockInfo* slot = vocabulary_->get(key);
    if (!slot) return std::nullopt;
    DiskBTreeIterator it(*this, slot);
    it.find(key);
    if (it.is_done()) return std::nullopt;
    if (cmp_keys(key, it.key()) == 0) return it;
    return std::nullopt;
}

std::optional<DiskBTreeIterator>
DiskBTreeReader::get_iterator(const std::string& key) const {
    std::vector<uint8_t> kv(key.begin(), key.end());
    return get_iterator(kv);
}

} // namespace galago
