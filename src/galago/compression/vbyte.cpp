// BSD License (http://www.galagosearch.org/license)
#include "galago/compression/vbyte.h"

#include <istream>
#include <ostream>
#include <stdexcept>

namespace galago {

// ── Decode from byte array ────────────────────────────────────────────────────

uint32_t vbyte_decode_u32(const uint8_t* data, size_t& offset) {
    uint32_t result = 0;
    for (int pos = 0; pos < 6; ++pos) {
        uint8_t b = data[offset++];
        if (b & 0x80u) {
            result |= static_cast<uint32_t>(b & 0x7fu) << (7 * pos);
            return result;
        }
        result |= static_cast<uint32_t>(b) << (7 * pos);
    }
    throw std::runtime_error("vbyte_decode_u32: unterminated sequence");
}

uint64_t vbyte_decode_u64(const uint8_t* data, size_t& offset) {
    uint64_t result = 0;
    for (int pos = 0; pos < 10; ++pos) {
        uint8_t b = data[offset++];
        if (b & 0x80u) {
            result |= static_cast<uint64_t>(b & 0x7fu) << (7 * pos);
            return result;
        }
        result |= static_cast<uint64_t>(b) << (7 * pos);
    }
    throw std::runtime_error("vbyte_decode_u64: unterminated sequence");
}

// ── Decode from stream ────────────────────────────────────────────────────────

uint32_t vbyte_decode_u32(std::istream& in) {
    uint32_t result = 0;
    for (int pos = 0; pos < 6; ++pos) {
        int raw = in.get();
        if (raw == EOF) throw std::runtime_error("vbyte_decode_u32: unexpected EOF");
        uint8_t b = static_cast<uint8_t>(raw);
        if (b & 0x80u) {
            result |= static_cast<uint32_t>(b & 0x7fu) << (7 * pos);
            return result;
        }
        result |= static_cast<uint32_t>(b) << (7 * pos);
    }
    throw std::runtime_error("vbyte_decode_u32: unterminated sequence");
}

uint64_t vbyte_decode_u64(std::istream& in) {
    uint64_t result = 0;
    for (int pos = 0; pos < 10; ++pos) {
        int raw = in.get();
        if (raw == EOF) throw std::runtime_error("vbyte_decode_u64: unexpected EOF");
        uint8_t b = static_cast<uint8_t>(raw);
        if (b & 0x80u) {
            result |= static_cast<uint64_t>(b & 0x7fu) << (7 * pos);
            return result;
        }
        result |= static_cast<uint64_t>(b) << (7 * pos);
    }
    throw std::runtime_error("vbyte_decode_u64: unterminated sequence");
}

// ── Encode to byte vector ─────────────────────────────────────────────────────

void vbyte_encode_u32(std::vector<uint8_t>& out, uint32_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<uint8_t>(v & 0x7fu));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v | 0x80u));
}

void vbyte_encode_u64(std::vector<uint8_t>& out, uint64_t v) {
    while (v >= 0x80u) {
        out.push_back(static_cast<uint8_t>(v & 0x7fu));
        v >>= 7;
    }
    out.push_back(static_cast<uint8_t>(v | 0x80u));
}

// ── Encode to stream ──────────────────────────────────────────────────────────

void vbyte_encode_u32(std::ostream& out, uint32_t v) {
    while (v >= 0x80u) {
        out.put(static_cast<char>(v & 0x7fu));
        v >>= 7;
    }
    out.put(static_cast<char>(v | 0x80u));
}

void vbyte_encode_u64(std::ostream& out, uint64_t v) {
    while (v >= 0x80u) {
        out.put(static_cast<char>(v & 0x7fu));
        v >>= 7;
    }
    out.put(static_cast<char>(v | 0x80u));
}

// ── Convenience ───────────────────────────────────────────────────────────────

std::vector<uint8_t> vbyte_encode_u32(uint32_t v) {
    std::vector<uint8_t> out;
    vbyte_encode_u32(out, v);
    return out;
}

} // namespace galago
