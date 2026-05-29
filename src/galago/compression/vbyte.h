#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of org.lemurproject.galago.utility.compression.VByte
//
// Galago VByte encoding: least-significant 7-bit group first.
// The *last* byte has its high bit set (0x80); all prior bytes have high bit clear.
// This is the inverse of many other VByte schemes.

#include <cstdint>
#include <cstddef>
#include <stdexcept>
#include <vector>
#include <iosfwd>

namespace galago {

// ── Decode ────────────────────────────────────────────────────────────────────

// Read a VByte-encoded uint32 from a byte array; advances *offset*.
uint32_t vbyte_decode_u32(const uint8_t* data, size_t& offset);

// Read a VByte-encoded uint64 from a byte array; advances *offset*.
uint64_t vbyte_decode_u64(const uint8_t* data, size_t& offset);

// Read a VByte-encoded uint32 from a stream (reads one byte at a time).
uint32_t vbyte_decode_u32(std::istream& in);

// Read a VByte-encoded uint64 from a stream.
uint64_t vbyte_decode_u64(std::istream& in);

// ── Encode ────────────────────────────────────────────────────────────────────

// Append a VByte-encoded uint32 to a byte vector.
void vbyte_encode_u32(std::vector<uint8_t>& out, uint32_t value);

// Append a VByte-encoded uint64 to a byte vector.
void vbyte_encode_u64(std::vector<uint8_t>& out, uint64_t value);

// Write a VByte-encoded uint32 to a stream.
void vbyte_encode_u32(std::ostream& out, uint32_t value);

// Write a VByte-encoded uint64 to a stream.
void vbyte_encode_u64(std::ostream& out, uint64_t value);

// Convenience: return encoded bytes for a uint32.
std::vector<uint8_t> vbyte_encode_u32(uint32_t value);

} // namespace galago
