#pragma once
// BSD License (http://www.galagosearch.org/license)
// Phase 5 index writing utilities.
//
// Encoding functions for Galago index binary formats, plus high-level
// write_names / write_lengths / write_postings helpers callable from Python.

#include "galago/btree/disk_btree_writer.h"
#include <cstdint>
#include <string>
#include <vector>
#include <map>

namespace galago {

// ── Posting list encoder ──────────────────────────────────────────────────────
// Encodes one term's posting list in PositionIndexCountSource format
// (options=HAS_MAXTF, no skips, no positions).
//
// postings: sorted (docid, count) pairs — docids must be strictly ascending.
// Returns the raw bytes to store as the B-tree value.

std::vector<uint8_t> encode_postings(
    const std::vector<std::pair<int64_t, int32_t>>& postings);

// ── Names writer ──────────────────────────────────────────────────────────────
// Write a `names` B-tree file: key=big-endian int64 docid, value=UTF-8 name.

void write_names(const std::string& path,
                 const std::vector<std::string>& names,
                 int64_t first_docid = 0);

// ── Lengths writer ────────────────────────────────────────────────────────────
// Write a `lengths` B-tree file.
// lengths[i] is the token count for document (first_docid + i).

void write_lengths(const std::string& path,
                   const std::vector<int32_t>& lengths,
                   int64_t first_docid = 0,
                   const std::string& field = "document");

// ── Postings writer ───────────────────────────────────────────────────────────
// Write a postings B-tree file given a sorted mapping term → [(docid, count)].
// term_postings must be sorted by term.

struct TermPostings {
    std::string term;
    std::vector<std::pair<int64_t, int32_t>> postings;  // (docid, count)
};

void write_postings_index(const std::string& path,
                          const std::vector<TermPostings>& sorted_terms,
                          int64_t total_docs,
                          int64_t collection_length);

// ── Positional postings encoder ───────────────────────────────────────────────
// Encode one term's positional posting list (positions embedded after counts).
// postings: sorted (docid, positions) pairs — docids must be strictly ascending.
// Returns raw bytes to store as the B-tree value.

std::vector<uint8_t> encode_positional_postings(
    const std::vector<std::pair<int64_t, std::vector<int32_t>>>& postings);

} // namespace galago
