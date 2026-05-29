// Tests for FileStream and DiskBTreeReader.
// The B-tree tests require a real Galago index file.
// They are gated on the GALAGO_TEST_INDEX environment variable.
//
// To run B-tree integration tests:
//   export GALAGO_TEST_INDEX=/path/to/index/postings.krovetz
//   ctest --test-dir build -R btree

#include <catch2/catch_test_macros.hpp>
#include "galago/io/file_stream.h"
#include "galago/btree/disk_btree_reader.h"
#include "galago/btree/btree_format.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <fstream>

using namespace galago;

// ── FileStream unit tests ─────────────────────────────────────────────────────

TEST_CASE("FileStream: write then read big-endian int", "[filestream]") {
    // Create a temp file with a known big-endian int.
    const char* tmpname = std::tmpnam(nullptr);
    {
        std::ofstream f(tmpname, std::ios::binary);
        uint8_t buf[] = {0x00, 0x00, 0x01, 0x00};  // big-endian 256
        f.write(reinterpret_cast<const char*>(buf), 4);
    }
    FileStream fs(tmpname);
    REQUIRE(fs.length() == 4);
    REQUIRE(fs.read_int() == 256);
    std::remove(tmpname);
}

TEST_CASE("FileStream: read_long big-endian", "[filestream]") {
    const char* tmpname = std::tmpnam(nullptr);
    {
        std::ofstream f(tmpname, std::ios::binary);
        // Write 0x0000000100000002 as two big-endian ints
        uint8_t buf[] = {0x00, 0x00, 0x00, 0x01,   // hi: 1
                          0x00, 0x00, 0x00, 0x02};  // lo: 2
        f.write(reinterpret_cast<const char*>(buf), 8);
    }
    FileStream fs(tmpname);
    int64_t v = fs.read_long();
    REQUIRE(v == static_cast<int64_t>(0x0000000100000002LL));
    std::remove(tmpname);
}

TEST_CASE("FileStream: sub_stream slicing", "[filestream]") {
    const char* tmpname = std::tmpnam(nullptr);
    {
        std::ofstream f(tmpname, std::ios::binary);
        uint8_t buf[] = {0x01, 0x02, 0x03, 0x04, 0x05};
        f.write(reinterpret_cast<const char*>(buf), 5);
    }
    FileStream fs(tmpname);
    FileStream sub = fs.absolute_sub_stream(1, 4);  // bytes [1,4) = {0x02, 0x03, 0x04}
    REQUIRE(sub.length() == 3);
    REQUIRE(sub.read_byte() == 0x02);
    REQUIRE(sub.read_byte() == 0x03);
    std::remove(tmpname);
}

TEST_CASE("FileStream: VByte round-trip via file", "[filestream]") {
    const char* tmpname = std::tmpnam(nullptr);
    {
        std::ofstream f(tmpname, std::ios::binary);
        // Manually write VByte(300) = 0x2C, 0x82
        uint8_t buf[] = {0x2C, 0x82};
        f.write(reinterpret_cast<const char*>(buf), 2);
    }
    FileStream fs(tmpname);
    REQUIRE(fs.read_vbyte_u32() == 300u);
    std::remove(tmpname);
}

// ── DiskBTreeReader integration tests ─────────────────────────────────────────

static std::string test_index_path() {
    const char* env = std::getenv("GALAGO_TEST_INDEX");
    return env ? std::string(env) : "";
}

TEST_CASE("DiskBTreeReader: detect non-btree file", "[btree]") {
    const char* tmpname = std::tmpnam(nullptr);
    {
        std::ofstream f(tmpname, std::ios::binary);
        f << "this is not a btree file";
    }
    REQUIRE_FALSE(DiskBTreeReader::is_btree(tmpname));
    std::remove(tmpname);
}

TEST_CASE("DiskBTreeReader: open and iterate real index", "[btree][integration]") {
    std::string path = test_index_path();
    if (path.empty()) {
        SKIP("Set GALAGO_TEST_INDEX to run B-tree integration tests");
    }

    REQUIRE(DiskBTreeReader::is_btree(path));

    DiskBTreeReader reader(path);
    REQUIRE_FALSE(reader.manifest_json().empty());

    auto it = reader.get_iterator();
    REQUIRE(it.has_value());

    int count = 0;
    while (!it->is_done() && count < 100) {
        REQUIRE_FALSE(it->key().empty());
        REQUIRE(it->value_length() >= 0);
        it->next_key();
        ++count;
    }
    REQUIRE(count > 0);
}

TEST_CASE("DiskBTreeReader: key lookup in real index", "[btree][integration]") {
    std::string path = test_index_path();
    if (path.empty()) {
        SKIP("Set GALAGO_TEST_INDEX to run B-tree integration tests");
    }

    DiskBTreeReader reader(path);

    // Get the first key from the index, then look it up by name.
    auto scan = reader.get_iterator();
    REQUIRE(scan.has_value());

    std::string first_key = scan->key_string();
    auto found = reader.get_iterator(first_key);
    REQUIRE(found.has_value());
    REQUIRE(found->key_string() == first_key);

    // Looking up a key that doesn't exist should return nullopt.
    auto missing = reader.get_iterator("\x00\x00this_key_cannot_exist");
    // Either nullopt, or found but key doesn't match (handled inside get_iterator).
    if (missing.has_value()) {
        REQUIRE(missing->key_string() != "\x00\x00this_key_cannot_exist");
    }
}
