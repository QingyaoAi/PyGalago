#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of CachedBufferDataStream / FileReadableBuffer
//
// FileStream: random-access file reader that exposes:
//   - Big-endian primitive reads (matching Java's DataInputStream)
//   - Sub-stream slicing (start..end byte range in the same file)
//   - VByte helpers

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>

namespace galago {

// ── FileStream ────────────────────────────────────────────────────────────────
// Represents a view [start, end) into an open file.
// All reads advance an internal position cursor.
// Big-endian for multi-byte primitives (matching Java DataInputStream).

class FileStream {
public:
    // Default-construct a null/empty stream (is_done() is always true).
    FileStream() : start_(0), end_(0), pos_(0), file_size_(0) {}

    // Open a file for reading.
    explicit FileStream(const std::string& path);

    // Create a sub-view of an already-open file.
    FileStream(std::shared_ptr<FILE> file, int64_t start, int64_t end);

    // Default copy/move
    FileStream(const FileStream&) = default;
    FileStream& operator=(const FileStream&) = default;
    FileStream(FileStream&&) = default;

    // ── Positioning ──────────────────────────────────────────────────────────

    int64_t position() const { return pos_ - start_; }   // relative to start
    int64_t length()   const { return end_ - start_; }
    bool    is_done()  const { return pos_ >= end_; }

    // Seek to absolute offset from *start* of this view.
    void seek(int64_t offset);

    // ── Primitive reads (big-endian) ─────────────────────────────────────────

    uint8_t  read_byte();
    int32_t  read_int();    // 4 bytes, big-endian signed
    int64_t  read_long();   // 8 bytes, big-endian signed
    int32_t  read_unsigned_short(); // 2 bytes, big-endian unsigned

    // Read exactly n bytes into buf.
    void read_fully(uint8_t* buf, size_t n);
    void read_fully(std::vector<uint8_t>& buf, size_t n);

    // ── VByte helpers ─────────────────────────────────────────────────────────

    uint32_t read_vbyte_u32();
    uint64_t read_vbyte_u64();

    // ── Sub-stream factory ────────────────────────────────────────────────────

    // Return a new FileStream covering [pos+offset, pos+offset+length) in the file.
    FileStream sub_stream(int64_t offset, int64_t length) const;

    // Return a FileStream from absolute file positions [abs_start, abs_end).
    FileStream absolute_sub_stream(int64_t abs_start, int64_t abs_end) const;

    int64_t file_size() const { return file_size_; }

private:
    std::shared_ptr<FILE> file_;
    int64_t start_;    // absolute start in file
    int64_t end_;      // absolute end in file (exclusive)
    int64_t pos_;      // current absolute position
    int64_t file_size_;

    // Read-ahead buffer — eliminates per-byte fseek overhead.
    static constexpr size_t BUFFER_SIZE = 65536;  // 64 KB
    std::vector<uint8_t>    buf_;
    int64_t                 buf_start_ = -1;   // absolute file pos of buf_[0]
    int64_t                 buf_size_  =  0;   // valid bytes in buf_

    void fill_buffer();
    uint8_t fetch_byte();
};

} // namespace galago
