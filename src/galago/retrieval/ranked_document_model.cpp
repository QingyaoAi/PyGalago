// BSD License (http://www.galagosearch.org/license)
#include "galago/retrieval/ranked_document_model.h"
#include "galago/retrieval/galago_count_iterator.h"
#include "galago/retrieval/bm25_iterator.h"
#include "galago/retrieval/score_combination_iterator.h"
#include "galago/retrieval/disjunction_iterator.h"
#include "galago/index/postings_reader.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <unordered_map>
#include <vector>

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

// ── ql_search helpers — TAAT (term-at-a-time) ────────────────────────────────
//
// For QL with Dirichlet smoothing, TAAT beats DAAT:
//   score(d) = Σ_t log((tf_t + μ·p_t)/(dl+μ))
//            = [Σ_t log(μ·p_t)] + [Σ_{tf>0} corr_t(d)] - n·log(dl+μ)
// We accumulate per-document corrections in one pass over each posting list,
// avoiding the O(union × n_terms) skip_to overhead of pure DAAT.

namespace {

struct QLTermInfo {
    double        mu_pt;
    double        log_mu_pt;
    PostingsIterator pit;
};

// Flat-array accumulator — allocated once per thread, reused across queries.
// Avoids unordered_map overhead; reset only touched slots after each query.
struct FlatAccumulator {
    std::vector<double>  corrections;   // indexed by doc_id
    std::vector<int64_t> touched;       // doc_ids written this query

    explicit FlatAccumulator(int64_t n_docs) : corrections(n_docs, 0.0) {
        touched.reserve(1 << 17);
    }

    inline void add(int64_t doc, double delta) {
        if (corrections[doc] == 0.0) touched.push_back(doc);
        corrections[doc] += delta;
    }

    void reset() {
        for (int64_t d : touched) corrections[d] = 0.0;
        touched.clear();
    }
};

// Thread-local accumulator — constructed lazily at first use.
static FlatAccumulator& get_accumulator(int64_t n_docs) {
    static thread_local FlatAccumulator acc(0);
    if (static_cast<int64_t>(acc.corrections.size()) < n_docs) {
        acc.corrections.assign(n_docs, 0.0);
        acc.touched.clear();
    }
    return acc;
}

static std::vector<ScoredDocument> ql_taat(
        std::vector<QLTermInfo>& terms,
        const LengthsSource&     lengths,
        double                   mu,
        int                      n)
{
    if (terms.empty()) return {};

    double sum_log_mu_pt = 0.0;
    for (const auto& ti : terms) sum_log_mu_pt += ti.log_mu_pt;
    const double n_terms = static_cast<double>(terms.size());

    const int64_t n_docs = lengths.stats().last_document + 1;
    FlatAccumulator& acc = get_accumulator(n_docs);

    // Pass 1: accumulate per-doc corrections (one linear scan per posting list).
    for (auto& ti : terms) {
        const double mu_pt   = ti.mu_pt;
        const double log_mpt = ti.log_mu_pt;
        auto& pit = ti.pit;
        while (!pit.is_done()) {
            acc.add(pit.doc_id(),
                    std::log(static_cast<double>(pit.count()) + mu_pt) - log_mpt);
            pit.next();
        }
    }

    // Pass 2: score touched docs and maintain a min-heap of top-n.
    using MinHeap = std::priority_queue<ScoredDocument,
                                        std::vector<ScoredDocument>,
                                        std::greater<ScoredDocument>>;
    MinHeap heap;

    for (int64_t doc : acc.touched) {
        int32_t dl   = lengths.length(doc);
        double score = sum_log_mu_pt + acc.corrections[doc]
                       - n_terms * std::log(static_cast<double>(dl) + mu);

        if (static_cast<int>(heap.size()) < n || heap.top().score < score) {
            if (static_cast<int>(heap.size()) >= n) heap.pop();
            heap.push({doc, score});
        }
    }

    acc.reset();

    std::vector<ScoredDocument> results;
    results.reserve(heap.size());
    while (!heap.empty()) { results.push_back(heap.top()); heap.pop(); }
    std::sort(results.begin(), results.end(),
              [](const ScoredDocument& a, const ScoredDocument& b) {
                  return a.score > b.score;
              });
    return results;
}

// Weighted variant (for WSDM-Int): w_t pre-normalised so Σw_t = 1.
static std::vector<ScoredDocument> ql_taat_weighted(
        std::vector<QLTermInfo>&   terms,
        const std::vector<double>& weights,
        const LengthsSource&       lengths,
        double                     mu,
        int                        n)
{
    if (terms.empty()) return {};

    double sum_w_log_mu_pt = 0.0;
    for (size_t i = 0; i < terms.size(); ++i)
        sum_w_log_mu_pt += weights[i] * terms[i].log_mu_pt;

    const int64_t n_docs = lengths.stats().last_document + 1;
    FlatAccumulator& acc = get_accumulator(n_docs);

    for (size_t i = 0; i < terms.size(); ++i) {
        const double w       = weights[i];
        const double mu_pt   = terms[i].mu_pt;
        const double log_mpt = terms[i].log_mu_pt;
        auto& pit = terms[i].pit;
        while (!pit.is_done()) {
            acc.add(pit.doc_id(),
                    w * (std::log(static_cast<double>(pit.count()) + mu_pt) - log_mpt));
            pit.next();
        }
    }

    using MinHeap = std::priority_queue<ScoredDocument,
                                        std::vector<ScoredDocument>,
                                        std::greater<ScoredDocument>>;
    MinHeap heap;

    for (int64_t doc : acc.touched) {
        int32_t dl   = lengths.length(doc);
        double score = sum_w_log_mu_pt + acc.corrections[doc]
                       - std::log(static_cast<double>(dl) + mu);

        if (static_cast<int>(heap.size()) < n || heap.top().score < score) {
            if (static_cast<int>(heap.size()) >= n) heap.pop();
            heap.push({doc, score});
        }
    }

    acc.reset();

    std::vector<ScoredDocument> results;
    results.reserve(heap.size());
    while (!heap.empty()) { results.push_back(heap.top()); heap.pop(); }
    std::sort(results.begin(), results.end(),
              [](const ScoredDocument& a, const ScoredDocument& b) {
                  return a.score > b.score;
              });
    return results;
}

} // anonymous namespace

// ── ql_search (DiskIndex overload) ───────────────────────────────────────────

std::vector<ScoredDocument> ql_search(DiskIndex&                      index,
                                       LengthsSource&                  lengths,
                                       const std::vector<std::string>& terms,
                                       const QLParams&                 params)
{
    PostingsReader* pr = index.postings_reader(params.postings_part);
    if (!pr) return {};

    const double C  = static_cast<double>(lengths.stats().collection_length);
    const double mu = params.mu;

    // Deduplicate terms, build QLTermInfo for each term in the index.
    std::vector<std::string> seen;
    std::vector<QLTermInfo> term_infos;
    for (const auto& term : terms) {
        if (std::find(seen.begin(), seen.end(), term) != seen.end()) continue;
        seen.push_back(term);

        auto s = pr->get_stats(term);
        if (!s || s->collection_count == 0) continue;
        auto pit_opt = index.get_postings(term, params.postings_part);
        if (!pit_opt) continue;

        double p_t   = static_cast<double>(s->collection_count) / C;
        double mu_pt = mu * p_t;
        term_infos.push_back({mu_pt, std::log(mu_pt), std::move(*pit_opt)});
    }

    return ql_taat(term_infos, lengths, mu, params.n);
}

// ── ql_search (path overload) ─────────────────────────────────────────────────

std::vector<ScoredDocument> ql_search(const std::string&              index_path,
                                       const std::vector<std::string>& terms,
                                       const QLParams&                 params)
{
    DiskIndex     index(index_path);
    LengthsSource lengths(index_path + "/lengths");
    return ql_search(index, lengths, terms, params);
}

// ── ql_search_weighted ────────────────────────────────────────────────────────

std::vector<ScoredDocument> ql_search_weighted(
        DiskIndex&                                         index,
        LengthsSource&                                     lengths,
        const std::vector<std::pair<std::string, double>>& weighted_terms,
        const QLParams&                                    params)
{
    PostingsReader* pr = index.postings_reader(params.postings_part);
    if (!pr) return {};

    const double C  = static_cast<double>(lengths.stats().collection_length);
    const double mu = params.mu;

    std::vector<QLTermInfo> term_infos;
    std::vector<double> raw_weights;

    for (const auto& [term, w] : weighted_terms) {
        if (w <= 0.0) continue;
        auto s = pr->get_stats(term);
        if (!s || s->collection_count == 0) continue;
        auto pit_opt = index.get_postings(term, params.postings_part);
        if (!pit_opt) continue;

        double p_t   = static_cast<double>(s->collection_count) / C;
        double mu_pt = mu * p_t;
        term_infos.push_back({mu_pt, std::log(mu_pt), std::move(*pit_opt)});
        raw_weights.push_back(w);
    }

    if (term_infos.empty()) return {};

    // Normalise weights to sum to 1.
    double total = std::accumulate(raw_weights.begin(), raw_weights.end(), 0.0);
    if (total <= 0.0) return {};
    std::vector<double> norm_weights(raw_weights.size());
    for (size_t i = 0; i < raw_weights.size(); ++i)
        norm_weights[i] = raw_weights[i] / total;

    return ql_taat_weighted(term_infos, norm_weights, lengths, mu, params.n);
}

} // namespace galago
