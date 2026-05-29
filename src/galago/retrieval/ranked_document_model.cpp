// BSD License (http://www.galagosearch.org/license)
#include "galago/retrieval/ranked_document_model.h"
#include "galago/retrieval/galago_count_iterator.h"
#include "galago/retrieval/bm25_iterator.h"
#include "galago/retrieval/score_combination_iterator.h"

#include <algorithm>
#include <queue>
#include <stdexcept>

namespace galago {

// ── DAAT top-k ────────────────────────────────────────────────────────────────
// Matches RankedDocumentModel.execute() from Java exactly.

std::vector<ScoredDocument> daat_top_k(ScoreIterator& root, int n) {
    // Min-heap: smallest score at top so we can easily evict the worst result.
    using MinHeap = std::priority_queue<ScoredDocument,
                                        std::vector<ScoredDocument>,
                                        std::greater<ScoredDocument>>;
    MinHeap heap;

    ScoringContext ctx;

    while (!root.is_done()) {
        ctx.document = root.current_candidate();
        root.sync_to(ctx.document);

        if (root.has_match(ctx)) {
            double sc = root.score(ctx);
            if (static_cast<int>(heap.size()) < n || heap.top().score < sc) {
                if (static_cast<int>(heap.size()) >= n) heap.pop();
                heap.push({ctx.document, sc});
            }
        }
        root.move_past(ctx.document);
    }

    std::vector<ScoredDocument> results;
    results.reserve(heap.size());
    while (!heap.empty()) {
        results.push_back(heap.top());
        heap.pop();
    }
    // Return in descending score order (rank 0 = best).
    std::sort(results.begin(), results.end(),
              [](const ScoredDocument& a, const ScoredDocument& b) {
                  return a.score > b.score;
              });
    return results;
}

// ── BM25 iterator tree factory ────────────────────────────────────────────────

// Build BM25 params for one term from the index statistics.
static BM25Iterator::Params make_bm25_params(const LengthsSource&  lengths,
                                              const PostingsReader& pr,
                                              const std::string&    term,
                                              const BM25Params&     cfg) {
    auto stats_opt = pr.get_stats(term);
    if (!stats_opt) return {};   // term not in index

    const LengthStats& ls = lengths.stats();
    BM25Iterator::Params p;
    p.b              = cfg.b;
    p.k              = cfg.k;
    p.avg_doc_length = ls.avg_length;
    p.doc_count      = ls.total_document_count;
    p.df             = stats_opt->document_count;
    p.max_tf         = stats_opt->max_tf;
    return p;
}

// ── bm25_search (DiskIndex overload) ─────────────────────────────────────────

std::vector<ScoredDocument> bm25_search(DiskIndex&                      index,
                                         LengthsSource&                  lengths,
                                         const std::vector<std::string>& terms,
                                         const BM25Params&               params) {
    const std::string& part = params.postings_part;

    // Collect the BM25 leaf iterators for terms that exist in the index.
    // We use manual memory management here: the BM25Iterator owns the
    // GalagoCountIterator; ScoreCombinationIterator borrows the BM25Iterators.
    std::vector<std::unique_ptr<BM25Iterator>> bm25_iters;
    std::vector<ScoreIterator*>                score_ptrs;

    for (const auto& term : terms) {
        auto pit_opt = index.get_postings(term, part);
        if (!pit_opt) continue;   // term not in index — skip

        PostingsReader* pr = index.postings_reader(part);

        BM25Iterator::Params p = pr
            ? make_bm25_params(lengths, *pr, term, params)
            : BM25Iterator::Params{};
        if (p.doc_count == 0) continue;

        auto count_iter = std::make_unique<GalagoCountIterator>(
            pr, term, std::move(*pit_opt));

        bm25_iters.push_back(std::make_unique<BM25Iterator>(
            std::move(count_iter), &lengths, p));
        score_ptrs.push_back(bm25_iters.back().get());
    }

    if (score_ptrs.empty()) return {};

    if (score_ptrs.size() == 1) {
        return daat_top_k(*score_ptrs[0], params.n);
    }

    // Multiple terms: wrap in #combine with uniform weights.
    ScoreCombinationIterator combine(score_ptrs);
    return daat_top_k(combine, params.n);
}

// ── bm25_search (path overload) ───────────────────────────────────────────────

std::vector<ScoredDocument> bm25_search(const std::string&              index_path,
                                         const std::vector<std::string>& terms,
                                         const BM25Params&               params) {
    DiskIndex    index(index_path);
    LengthsSource lengths(index_path + "/lengths");
    return bm25_search(index, lengths, terms, params);
}

// ── bm25_search_weighted ──────────────────────────────────────────────────────

std::vector<ScoredDocument> bm25_search_weighted(
        DiskIndex&                                         index,
        LengthsSource&                                     lengths,
        const std::vector<std::pair<std::string, double>>& weighted_terms,
        const BM25Params&                                  params) {

    const std::string& part = params.postings_part;
    PostingsReader*    pr   = index.postings_reader(part);

    std::vector<std::unique_ptr<BM25Iterator>> bm25_iters;
    std::vector<ScoreIterator*>                score_ptrs;
    std::vector<double>                        weights;

    for (const auto& [term, w] : weighted_terms) {
        if (w <= 0.0) continue;
        auto pit_opt = index.get_postings(term, part);
        if (!pit_opt) continue;

        BM25Iterator::Params p = pr
            ? make_bm25_params(lengths, *pr, term, params)
            : BM25Iterator::Params{};
        if (p.doc_count == 0) continue;

        auto count_iter = std::make_unique<GalagoCountIterator>(
            pr, term, std::move(*pit_opt));

        bm25_iters.push_back(std::make_unique<BM25Iterator>(
            std::move(count_iter), &lengths, p));
        score_ptrs.push_back(bm25_iters.back().get());
        weights.push_back(w);
    }

    if (score_ptrs.empty()) return {};
    if (score_ptrs.size() == 1) return daat_top_k(*score_ptrs[0], params.n);

    ScoreCombinationIterator combine(score_ptrs, weights);
    return daat_top_k(combine, params.n);
}

} // namespace galago
