#pragma once
// BSD License (http://www.galagosearch.org/license)
// Port of DiskIndex.java (top-level index directory accessor).
//
// A Galago index directory typically contains:
//   names           — docid → document name (B-tree, key=int64 big-endian)
//   names.reverse   — document name → docid
//   lengths         — field → length array (B-tree, key=field name)
//   postings        — term → posting list (count index, no positions)
//   postings.krovetz— same, with Krovetz-stemmed terms
//   corpus          — raw document store (separate format, not yet ported)

#include "galago/index/names_reader.h"
#include "galago/index/lengths_reader.h"
#include "galago/index/postings_reader.h"

#include <map>
#include <memory>
#include <optional>
#include <string>

namespace galago {

class DiskIndex {
public:
    // Open an index directory. Only parts that are present are loaded.
    explicit DiskIndex(const std::string& index_path);

    // ── Names ─────────────────────────────────────────────────────────────────
    bool has_names() const { return names_ != nullptr; }
    std::string get_name(int64_t docid) const;

    // ── Lengths ───────────────────────────────────────────────────────────────
    bool has_lengths() const { return lengths_ != nullptr; }
    int32_t  get_length(int64_t docid) const;
    LengthStats get_length_stats(const std::string& field = "document") const;
    int64_t  total_documents() const;

    // ── Postings ──────────────────────────────────────────────────────────────
    // Returns an iterator for the given postings part and term.
    // part: "postings" | "postings.krovetz" (default)
    bool has_postings(const std::string& part = "postings.krovetz") const;
    std::optional<PostingsIterator> get_postings(const std::string& term,
                                                  const std::string& part = "postings.krovetz") const;

    const std::string& path() const { return path_; }

private:
    std::string path_;

    std::unique_ptr<NamesReader>   names_;
    std::unique_ptr<LengthsReader> lengths_;

    // Multiple postings parts may coexist.
    mutable std::map<std::string, std::unique_ptr<PostingsReader>> postings_parts_;

    void try_open_part(const std::string& name);
    PostingsReader* get_postings_reader(const std::string& part) const;
};

} // namespace galago
