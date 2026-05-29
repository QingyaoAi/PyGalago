// BSD License (http://www.galagosearch.org/license)
#include "galago/retrieval/lengths_source.h"
#include <stdexcept>

namespace galago {

// ── Construction ──────────────────────────────────────────────────────────────

LengthsSource::LengthsSource(const std::string& path, const std::string& field) {
    LengthsReader reader(path);
    load(reader, field);
}

LengthsSource::LengthsSource(const LengthsReader& reader, const std::string& field) {
    load(reader, field);
}

void LengthsSource::load(const LengthsReader& reader, const std::string& field) {
    stats_ = reader.get_stats(field);
    first_doc_   = stats_.first_document;
    last_doc_    = stats_.last_document;
    current_doc_ = first_doc_;

    if (last_doc_ < first_doc_) return;  // empty index

    size_t n = static_cast<size_t>(last_doc_ - first_doc_ + 1);
    lengths_.resize(n, 0);

    reader.for_each([&](int64_t docid, int32_t len) {
        size_t idx = static_cast<size_t>(docid - first_doc_);
        if (idx < n) lengths_[idx] = len;
        return true;
    }, field);
}

// ── length lookup ─────────────────────────────────────────────────────────────

int32_t LengthsSource::length(int64_t docid) const {
    if (docid < first_doc_ || docid > last_doc_) return 0;
    return lengths_[static_cast<size_t>(docid - first_doc_)];
}

} // namespace galago
