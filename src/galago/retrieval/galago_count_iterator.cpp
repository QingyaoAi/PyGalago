// BSD License (http://www.galagosearch.org/license)
#include "galago/retrieval/galago_count_iterator.h"
#include <stdexcept>

namespace galago {

GalagoCountIterator::GalagoCountIterator(PostingsReader* pr,
                                          std::string term,
                                          PostingsIterator pit)
    : reader_(pr), term_(std::move(term)), pit_(std::move(pit))
{}

int64_t GalagoCountIterator::current_candidate() const {
    return pit_.is_done() ? DONE : pit_.doc_id();
}

bool GalagoCountIterator::has_match(const ScoringContext& ctx) const {
    return !pit_.is_done() && pit_.doc_id() == ctx.document;
}

int32_t GalagoCountIterator::count(const ScoringContext& ctx) const {
    if (!has_match(ctx)) return 0;
    return pit_.count();
}

void GalagoCountIterator::sync_to(int64_t doc) {
    if (!pit_.is_done() && pit_.doc_id() < doc) {
        pit_.skip_to(doc);
    }
}

void GalagoCountIterator::reset() {
    if (!reader_) return;
    auto opt = reader_->get_postings(term_);
    if (opt) pit_ = std::move(*opt);
}

} // namespace galago
