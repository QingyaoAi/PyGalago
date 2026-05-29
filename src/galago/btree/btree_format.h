#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskBTreeFormat.java

#include <cstdint>

namespace galago {

// Magic number that appears at the end of every Galago B-tree file.
// Footer layout (last 28 bytes of file):
//   [fileLen-28] vocabulary_offset : int64 big-endian
//   [fileLen-20] manifest_offset   : int64 big-endian
//   [fileLen-12] block_size        : int32 big-endian
//   [fileLen-8]  MAGIC             : int64 big-endian
static constexpr int64_t  BTREE_MAGIC  = 0x1a2b3c4d5e6f7a8dLL;
static constexpr int64_t  FOOTER_SIZE  = 8 + 8 + 4 + 8;  // = 28 bytes

} // namespace galago
