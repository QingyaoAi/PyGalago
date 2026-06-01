// BSD License (http://www.galagosearch.org/license)
#include "galago/io/file_stream.h"
#include "galago/compression/vbyte.h"

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace galago {

// ── Construction ──────────────────────────────────────────────────────────────

FileStream::FileStream(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) {
        throw std::runtime_error("Cannot open file: " + path);
    }
    // Custom deleter so the file is closed when the last FileStream sharing it
    // goes out of scope.
    file_ = std::shared_ptr<FILE>(f, [](FILE* fp) { if (fp) std::fclose(fp); });

    std::fseek(f, 0, SEEK_END);
    file_size_ = static_cast<int64_t>(std::ftell(f));

    start_ = 0;
    end_   = file_size_;
    pos_   = 0;
}

FileStream::FileStream(std::shared_ptr<FILE> file, int64_t start, int64_t end)
    : file_(std::move(file)), start_(start), end_(end), pos_(start)
{
    // Determine file size
    std::fseek(file_.get(), 0, SEEK_END);
    file_size_ = static_cast<int64_t>(std::ftell(file_.get()));
}

// ── Positioning ───────────────────────────────────────────────────────────────

void FileStream::seek(int64_t offset) {
    int64_t abs = start_ + offset;
    if (abs < start_ || abs > end_) {
        throw std::out_of_range("FileStream::seek out of range");
    }
    pos_ = abs;
    // Don't invalidate the buffer — if the new pos is within it, we reuse it.
}

// ── Internal byte fetch with 64 KB read-ahead buffer ─────────────────────────

void FileStream::fill_buffer() {
    if (buf_.empty()) buf_.resize(BUFFER_SIZE);
    buf_start_ = pos_;
    int64_t to_read = std::min(end_ - pos_, static_cast<int64_t>(BUFFER_SIZE));
    if (std::fseek(file_.get(), static_cast<long>(pos_), SEEK_SET) != 0)
        throw std::runtime_error("FileStream: fseek failed");
    buf_size_ = static_cast<int64_t>(
        std::fread(buf_.data(), 1, static_cast<size_t>(to_read), file_.get()));
}

uint8_t FileStream::fetch_byte() {
    if (pos_ >= end_)
        throw std::runtime_error("FileStream: read past end of stream");
    int64_t off = pos_ - buf_start_;
    if (off < 0 || off >= buf_size_)
        fill_buffer(), off = 0;
    ++pos_;
    return buf_[static_cast<size_t>(off)];
}

// ── Primitive reads ───────────────────────────────────────────────────────────

uint8_t FileStream::read_byte() {
    return fetch_byte();
}

int32_t FileStream::read_int() {
    uint8_t buf[4];
    read_fully(buf, 4);
    return static_cast<int32_t>(
        (static_cast<uint32_t>(buf[0]) << 24) |
        (static_cast<uint32_t>(buf[1]) << 16) |
        (static_cast<uint32_t>(buf[2]) <<  8) |
        (static_cast<uint32_t>(buf[3]))
    );
}

int64_t FileStream::read_long() {
    // Java reads two big-endian ints and combines
    uint32_t hi = static_cast<uint32_t>(read_int());
    uint32_t lo = static_cast<uint32_t>(read_int());
    return (static_cast<int64_t>(hi) << 32) | static_cast<int64_t>(lo);
}

int32_t FileStream::read_unsigned_short() {
    uint8_t buf[2];
    read_fully(buf, 2);
    return (static_cast<int32_t>(buf[0]) << 8) | static_cast<int32_t>(buf[1]);
}

void FileStream::read_fully(uint8_t* buf, size_t n) {
    if (pos_ + static_cast<int64_t>(n) > end_)
        throw std::runtime_error("FileStream::read_fully: request exceeds stream bounds");
    // Serve as much as possible from the read-ahead buffer, then fread the rest.
    int64_t off = pos_ - buf_start_;
    int64_t buffered = (off >= 0 && off < buf_size_) ? buf_size_ - off : 0;
    if (buffered > 0) {
        size_t from_buf = std::min(static_cast<size_t>(buffered), n);
        std::memcpy(buf, buf_.data() + static_cast<size_t>(off), from_buf);
        pos_ += static_cast<int64_t>(from_buf);
        buf  += from_buf;
        n    -= from_buf;
    }
    if (n > 0) {
        if (std::fseek(file_.get(), static_cast<long>(pos_), SEEK_SET) != 0)
            throw std::runtime_error("FileStream: fseek failed");
        size_t got = std::fread(buf, 1, n, file_.get());
        if (got != n)
            throw std::runtime_error("FileStream::read_fully: short read");
        pos_ += static_cast<int64_t>(n);
        buf_start_ = -1;  // buffer is now stale
        buf_size_  = 0;
    }
}

void FileStream::read_fully(std::vector<uint8_t>& buf, size_t n) {
    buf.resize(n);
    if (n > 0) read_fully(buf.data(), n);
}

// ── VByte helpers ─────────────────────────────────────────────────────────────

uint32_t FileStream::read_vbyte_u32() {
    uint32_t result = 0;
    for (int shift = 0; shift < 35; shift += 7) {
        uint8_t b = fetch_byte();
        if (b & 0x80u) {
            result |= static_cast<uint32_t>(b & 0x7fu) << shift;
            return result;
        }
        result |= static_cast<uint32_t>(b) << shift;
    }
    throw std::runtime_error("vbyte_u32: unterminated sequence");
}

uint64_t FileStream::read_vbyte_u64() {
    uint64_t result = 0;
    for (int shift = 0; shift < 70; shift += 7) {
        uint8_t b = fetch_byte();
        if (b & 0x80u) {
            result |= static_cast<uint64_t>(b & 0x7fu) << shift;
            return result;
        }
        result |= static_cast<uint64_t>(b) << shift;
    }
    throw std::runtime_error("vbyte_u64: unterminated sequence");
}

// ── Sub-stream factory ────────────────────────────────────────────────────────

FileStream FileStream::sub_stream(int64_t offset, int64_t length) const {
    int64_t abs_start = start_ + offset;
    int64_t abs_end   = abs_start + length;
    if (abs_start < start_ || abs_end > end_) {
        throw std::out_of_range("FileStream::sub_stream: range out of bounds");
    }
    return FileStream(file_, abs_start, abs_end);
}

FileStream FileStream::absolute_sub_stream(int64_t abs_start, int64_t abs_end) const {
    return FileStream(file_, abs_start, abs_end);
}

} // namespace galago
