// BSD License (http://www.galagosearch.org/license)
#include "galago/btree/disk_btree_writer.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <stdexcept>
#include <sstream>

namespace galago {

// ── Big-endian output helpers ─────────────────────────────────────────────────

void DiskBTreeWriter::write_be_i32(std::ostream& o, int32_t v) {
    uint8_t buf[4];
    buf[0] = static_cast<uint8_t>((static_cast<uint32_t>(v) >> 24) & 0xFF);
    buf[1] = static_cast<uint8_t>((static_cast<uint32_t>(v) >> 16) & 0xFF);
    buf[2] = static_cast<uint8_t>((static_cast<uint32_t>(v) >>  8) & 0xFF);
    buf[3] = static_cast<uint8_t>( static_cast<uint32_t>(v)        & 0xFF);
    o.write(reinterpret_cast<const char*>(buf), 4);
}

void DiskBTreeWriter::write_be_i64(std::ostream& o, int64_t v) {
    write_be_i32(o, static_cast<int32_t>(static_cast<uint64_t>(v) >> 32));
    write_be_i32(o, static_cast<int32_t>(static_cast<uint64_t>(v) & 0xFFFFFFFFULL));
}

// ── Construction ──────────────────────────────────────────────────────────────

DiskBTreeWriter::DiskBTreeWriter(const std::string& path, int block_size)
    : block_size_(block_size)
{
    out_.open(path, std::ios::binary | std::ios::out | std::ios::trunc);
    if (!out_) throw std::runtime_error("Cannot open for writing: " + path);
}

// ── Key helpers ───────────────────────────────────────────────────────────────

int DiskBTreeWriter::prefix_overlap(const std::vector<uint8_t>& a,
                                     const std::vector<uint8_t>& b) const {
    int max_ov = static_cast<int>(std::min(a.size(), b.size()));
    max_ov = std::min(max_ov, block_size_ - 1);  // keyOverlap limit
    for (int i = 0; i < max_ov; ++i) {
        if (a[i] != b[i]) return i;
    }
    return max_ov;
}

std::vector<uint8_t> DiskBTreeWriter::increment_key(const std::vector<uint8_t>& key) {
    std::vector<uint8_t> r(key);
    int i = static_cast<int>(r.size()) - 1;
    while (i >= 0 && r[i] == 0x7F) --i;  // Byte.MAX_VALUE in Java = 127 = 0x7F
    if (i >= 0) {
        r[i]++;
    } else {
        r.push_back(0x00);
    }
    return r;
}

// ── Size estimation ───────────────────────────────────────────────────────────

int64_t DiskBTreeWriter::pending_data_size() const {
    int64_t total = 8; // int64 key count in header
    for (const auto& e : pending_) {
        total += static_cast<int64_t>(e.key.size());
        total += 4; // key length overhead (2 overlap + 2 length)
        total += static_cast<int64_t>(e.value.size());
    }
    return total;
}

bool DiskBTreeWriter::needs_flush(const std::vector<uint8_t>& value) const {
    int64_t extra = 4; // per-entry overhead
    int64_t new_total = pending_data_size()
                      + static_cast<int64_t>(pending_.empty() ? 0 : prev_key_.size())
                      + static_cast<int64_t>(value.size())
                      + extra;
    return new_total >= block_size_;
}

// ── Block writing ─────────────────────────────────────────────────────────────

std::vector<uint8_t> DiskBTreeWriter::make_block_header() const {
    std::vector<uint8_t> hdr;

    // int64 key count (big-endian)
    int64_t kc = static_cast<int64_t>(pending_.size());
    uint64_t ukc = static_cast<uint64_t>(kc);
    for (int i = 7; i >= 0; --i) {
        hdr.push_back(static_cast<uint8_t>((ukc >> (8 * i)) & 0xFF));
    }

    // Compute total value data length for remaining-bytes encoding
    int64_t total_value_len = 0;
    for (const auto& e : pending_) {
        total_value_len += static_cast<int64_t>(e.value.size());
    }

    int64_t accumulated = 0;
    const std::vector<uint8_t>* prev = nullptr;
    for (size_t i = 0; i < pending_.size(); ++i) {
        const auto& key = pending_[i].key;
        int64_t remaining = total_value_len - accumulated
                          - static_cast<int64_t>(pending_[i].value.size());

        if (i == 0) {
            vbyte_encode_u32(hdr, static_cast<uint32_t>(key.size()));
            hdr.insert(hdr.end(), key.begin(), key.end());
        } else {
            int overlap = prefix_overlap(*prev, key);
            vbyte_encode_u32(hdr, static_cast<uint32_t>(overlap));
            vbyte_encode_u32(hdr, static_cast<uint32_t>(key.size()));
            hdr.insert(hdr.end(), key.begin() + overlap, key.end());
        }
        vbyte_encode_u32(hdr, static_cast<uint32_t>(remaining));
        accumulated += static_cast<int64_t>(pending_[i].value.size());
        prev = &key;
    }
    return hdr;
}

void DiskBTreeWriter::flush_block() {
    if (pending_.empty()) return;

    std::vector<uint8_t> header = make_block_header();

    int64_t block_start = file_position_;
    int32_t header_len  = static_cast<int32_t>(header.size());

    // Record vocabulary entry
    vocabulary_.push_back({pending_[0].key, block_start, header_len});

    // Write header
    out_.write(reinterpret_cast<const char*>(header.data()), header.size());
    file_position_ += static_cast<int64_t>(header.size());

    // Write value data
    for (const auto& e : pending_) {
        out_.write(reinterpret_cast<const char*>(e.value.data()), e.value.size());
        file_position_ += static_cast<int64_t>(e.value.size());
    }

    ++block_count_;
    pending_.clear();
    buffered_bytes_ = 0;
}

// ── Public: add ───────────────────────────────────────────────────────────────

void DiskBTreeWriter::add(const std::vector<uint8_t>& key,
                           const std::vector<uint8_t>& value) {
    if (!prev_key_.empty()) {
        if (key <= prev_key_) {
            throw std::runtime_error("DiskBTreeWriter: keys must be strictly ascending");
        }
    }
    if (needs_flush(value)) {
        flush_block();
    }
    pending_.push_back({key, value});
    prev_key_ = key;
    last_key_ = key;
    ++key_count_;
}

void DiskBTreeWriter::add(const std::string& key, const std::vector<uint8_t>& value) {
    add(std::vector<uint8_t>(key.begin(), key.end()), value);
}

// ── Public: close ─────────────────────────────────────────────────────────────

void DiskBTreeWriter::close(const std::string& manifest_json) {
    flush_block();

    std::vector<uint8_t> final_key = increment_key(last_key_);

    // ── Vocabulary section ────────────────────────────────────────────────────
    // Layout:
    //   int32 finalKeyLength + finalKey bytes
    //   per-block: VByte(keyLen) + key + VByte(blockOffset) + VByte(headerLen)

    std::vector<uint8_t> vocab_data;

    // Final key
    uint32_t fk_len = static_cast<uint32_t>(final_key.size());
    // Write as big-endian int32 to vocab_data
    vocab_data.push_back(static_cast<uint8_t>(fk_len >> 24));
    vocab_data.push_back(static_cast<uint8_t>(fk_len >> 16));
    vocab_data.push_back(static_cast<uint8_t>(fk_len >>  8));
    vocab_data.push_back(static_cast<uint8_t>(fk_len));
    vocab_data.insert(vocab_data.end(), final_key.begin(), final_key.end());

    for (const auto& ve : vocabulary_) {
        vbyte_encode_u32(vocab_data, static_cast<uint32_t>(ve.first_key.size()));
        vocab_data.insert(vocab_data.end(), ve.first_key.begin(), ve.first_key.end());
        vbyte_encode_u64(vocab_data, static_cast<uint64_t>(ve.offset));
        vbyte_encode_u32(vocab_data, static_cast<uint32_t>(ve.header_length));
    }

    // ── Manifest ──────────────────────────────────────────────────────────────
    std::vector<uint8_t> manifest_bytes(manifest_json.begin(), manifest_json.end());

    // Record offsets
    int64_t vocab_offset    = file_position_;
    int64_t manifest_offset = vocab_offset + static_cast<int64_t>(vocab_data.size());

    out_.write(reinterpret_cast<const char*>(vocab_data.data()),
               static_cast<std::streamsize>(vocab_data.size()));
    out_.write(reinterpret_cast<const char*>(manifest_bytes.data()),
               static_cast<std::streamsize>(manifest_bytes.size()));

    // ── Footer: vocab_offset(8) + manifest_offset(8) + blockSize(4) + MAGIC(8) ──
    write_be_i64(out_, vocab_offset);
    write_be_i64(out_, manifest_offset);
    write_be_i32(out_, block_size_);
    write_be_i64(out_, BTREE_MAGIC);

    out_.flush();
    out_.close();
}

} // namespace galago
