// Tests for galago::vbyte — mirrors Java VByte semantics exactly.
#include <catch2/catch_test_macros.hpp>
#include "galago/compression/vbyte.h"

#include <sstream>
#include <vector>
#include <cstdint>

using namespace galago;

// ── Encode → decode round-trips ───────────────────────────────────────────────

TEST_CASE("VByte u32 single-byte values", "[vbyte]") {
    // Values 0-127 encode as a single byte with high bit set.
    for (uint32_t v = 0; v < 128; ++v) {
        auto enc = vbyte_encode_u32(v);
        REQUIRE(enc.size() == 1);
        REQUIRE((enc[0] & 0x80u) != 0);
        size_t off = 0;
        REQUIRE(vbyte_decode_u32(enc.data(), off) == v);
        REQUIRE(off == 1);
    }
}

TEST_CASE("VByte u32 two-byte values", "[vbyte]") {
    // 128 requires 2 bytes.
    auto enc = vbyte_encode_u32(128);
    REQUIRE(enc.size() == 2);
    size_t off = 0;
    REQUIRE(vbyte_decode_u32(enc.data(), off) == 128u);

    enc = vbyte_encode_u32(16383);
    REQUIRE(enc.size() == 2);
    off = 0;
    REQUIRE(vbyte_decode_u32(enc.data(), off) == 16383u);
}

TEST_CASE("VByte u32 boundary values", "[vbyte]") {
    std::vector<uint32_t> cases = {0, 1, 127, 128, 16383, 16384, 2097151,
                                    2097152, 268435455, 268435456,
                                    0xFFFFFFFFu};
    for (uint32_t v : cases) {
        auto enc = vbyte_encode_u32(v);
        size_t off = 0;
        REQUIRE(vbyte_decode_u32(enc.data(), off) == v);
        REQUIRE(off == enc.size());
    }
}

TEST_CASE("VByte u32 stream round-trip", "[vbyte]") {
    std::ostringstream oss;
    std::vector<uint32_t> vals = {0, 1, 127, 128, 1000000, 0xFFFFFFFFu};
    for (uint32_t v : vals) vbyte_encode_u32(oss, v);

    std::istringstream iss(oss.str());
    for (uint32_t v : vals) {
        REQUIRE(vbyte_decode_u32(iss) == v);
    }
}

TEST_CASE("VByte u64 large values", "[vbyte]") {
    std::vector<uint64_t> cases = {
        0ULL, 1ULL, 127ULL, 128ULL,
        0xFFFFFFFFULL,
        0x1FFFFFFFFULL,
        0x7FFFFFFFFFFFFFFFULL,
        0xFFFFFFFFFFFFFFFFULL
    };
    for (uint64_t v : cases) {
        std::vector<uint8_t> enc;
        vbyte_encode_u64(enc, v);
        size_t off = 0;
        REQUIRE(vbyte_decode_u64(enc.data(), off) == v);
        REQUIRE(off == enc.size());
    }
}

TEST_CASE("VByte u64 stream round-trip", "[vbyte]") {
    std::ostringstream oss;
    std::vector<uint64_t> vals = {0, 128, 0xFFFFFFFFULL, 0x1FFFFFFFFULL};
    for (uint64_t v : vals) vbyte_encode_u64(oss, v);

    std::istringstream iss(oss.str());
    for (uint64_t v : vals) {
        REQUIRE(vbyte_decode_u64(iss) == v);
    }
}

// ── Java compatibility: specific known-good byte sequences ────────────────────
// These byte sequences were verified against the Java VByte.java output.

TEST_CASE("VByte Java compatibility", "[vbyte][compat]") {
    // Value 300: 300 = 0b1_0010_1100
    // Groups of 7: 0101100 = 44, 0000010 = 2
    // Encoded (Java LSB first, last byte high-bit set): 0x2C, 0x82
    {
        uint8_t expected[] = {0x2C, 0x82};
        size_t off = 0;
        REQUIRE(vbyte_decode_u32(expected, off) == 300u);

        auto enc = vbyte_encode_u32(300);
        REQUIRE(enc.size() == 2);
        REQUIRE(enc[0] == 0x2C);
        REQUIRE(enc[1] == 0x82);
    }

    // Value 1: single byte 0x81
    {
        uint8_t expected[] = {0x81};
        size_t off = 0;
        REQUIRE(vbyte_decode_u32(expected, off) == 1u);
        auto enc = vbyte_encode_u32(1);
        REQUIRE(enc.size() == 1);
        REQUIRE(enc[0] == 0x81);
    }
}
