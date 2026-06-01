#!/usr/bin/env python3
# BSD License (http://www.galagosearch.org/license)
"""Robust04 retrieval experiment: BM25, QL, SDM, WSDM, RM3.

Matches the experimental setup of Huston & Croft (CIKM 2014):
  - Porter2-stemmed index (postings.porter)
  - INQUERY stopword removal on queries
  - μ=2500 Dirichlet prior for QL-based models
  - BM25 b=0.75, k1=1.2

Paper targets (Table 7, Robust-04 collection):
  Title queries:        QL=0.252, BM25=0.254, SDM=0.263, WSDM-Int=0.269
  Description queries:  QL=0.244, BM25=0.237, SDM=0.258, WSDM-Int=0.278

Note on SDM: the Robust04 index contains positional posting lists, but
PyGalago's current C++ scorer uses count-only retrieval.  The Python-side
SDM below provides the correct ordered/unordered bigram scoring using
the positional postings API.

Usage
-----
    python scripts/robust04_experiment.py [--output results/robust04.md]
    python scripts/robust04_experiment.py --queries descs --output results/robust04_descs.md
"""
from __future__ import annotations

import argparse
import bisect
import math
import os
import random
import sys
import time
from array import array as _array
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

# ── PyGalago imports ──────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygalago._galago as g
from pygalago.parse.stemmer    import get_stemmer
from pygalago.parse.tokenizer  import tokenize_string
from pygalago.retrieval        import Retrieval
from pygalago.eval             import read_qrels, evaluate
from pygalago.eval.run         import write_run, Run, RankedDoc

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX        = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04.index"
PART         = "postings.porter"          # Porter2-stemmed (matches paper)
STEMMER_NAME = "porter"                   # must match PART
_QUERIES_DIR = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries"
TITLES_TSV   = os.path.join(_QUERIES_DIR, "rob04.titles.tsv")
DESCS_TSV    = os.path.join(_QUERIES_DIR, "rob04.descs.tsv")
QRELS_FILE   = os.path.join(_QUERIES_DIR, "robust04.qrels")

N         = 1000    # documents to retrieve per query
MU        = 2500    # Dirichlet prior (Galago / paper default)
BM25_B    = 0.75
BM25_K    = 1.2
SDM_UNI   = 0.85    # SDM unigram weight (paper default)
SDM_OD    = 0.10    # SDM ordered-window weight
SDM_UW    = 0.05    # SDM unordered-window weight
SDM_WIN   = 8       # SDM unordered window size (4 × term count; use fixed 8)
FB_DOCS   = 10      # RM3 feedback documents
FB_TERMS  = 20      # RM3 expansion terms
RM3_LAM   = 0.6     # RM3 interpolation (original query weight)
RM3_VOCAB = 500     # expansion vocabulary size for RM3

METRICS = ["map", "ndcg@20", "p@20"]

# ── INQUERY stopwords (Callan et al. 1994, standard 418-word list) ────────────
# fmt: off
INQUERY_STOPS = frozenset({
    "a", "about", "above", "according", "across", "after", "afterwards",
    "again", "against", "albeit", "all", "almost", "alone", "along",
    "already", "also", "although", "always", "am", "among", "amongst",
    "an", "and", "another", "any", "anybody", "anyhow", "anyone",
    "anything", "anyway", "anywhere", "apart", "are", "around", "as",
    "at", "av", "be", "became", "because", "become", "becomes",
    "becoming", "been", "before", "beforehand", "behind", "being",
    "below", "beside", "besides", "between", "beyond", "both",
    "but", "by", "can", "cannot", "canst", "certain", "cf", "choose",
    "contrariwise", "cos", "could", "couldn't", "dare", "daren't",
    "definitely", "despite", "did", "didn't", "different", "directly",
    "do", "does", "doesn't", "doing", "done", "don't", "down",
    "during", "e", "each", "eg", "either", "else", "elsewhere",
    "enough", "etc", "even", "ever", "every", "everybody", "everyone",
    "everything", "everywhere", "except", "exactly", "far", "few",
    "ff", "fifth", "first", "following", "for", "former", "formerly",
    "forth", "from", "further", "furthermore", "get", "given", "go",
    "got", "h", "had", "hadn't", "has", "hasn't", "have", "haven't",
    "having", "he", "her", "here", "hereabouts", "hereafter", "hereby",
    "herein", "hereinafter", "heretofore", "hereunder", "hereupon",
    "herewith", "him", "himself", "his", "how", "however", "i", "ie",
    "if", "in", "indeed", "inside", "instead", "into", "is", "isn't",
    "it", "its", "itself", "just", "kind", "kg", "km", "last",
    "latter", "latterly", "less", "lest", "let", "like", "little",
    "lots", "many", "may", "maybe", "me", "meantime", "meanwhile",
    "might", "moreover", "most", "mostly", "more", "mr", "mrs",
    "much", "my", "myself", "namely", "needn't", "neither", "never",
    "nevertheless", "next", "no", "nobody", "none", "noone", "nothing",
    "notwithstanding", "now", "nowhere", "of", "off", "often", "ok",
    "on", "once", "one", "only", "onto", "or", "other", "others",
    "otherwise", "ought", "our", "ours", "ourselves", "out", "outside",
    "over", "own", "per", "perhaps", "please", "rather", "re",
    "really", "regarding", "same", "sans", "self", "several", "should",
    "shouldn't", "since", "so", "some", "somebody", "somehow",
    "someone", "something", "sometime", "sometimes", "somewhere",
    "still", "such", "than", "that", "the", "thee", "their", "theirs",
    "them", "themselves", "then", "thence", "there", "thereabouts",
    "thereafter", "thereby", "therfore", "therefore", "therein",
    "these", "they", "this", "those", "thou", "though", "through",
    "throughout", "thru", "thus", "thy", "till", "to", "together",
    "too", "toward", "towards", "under", "unless", "until", "up",
    "upon", "us", "very", "via", "vs", "was", "wasn't", "we",
    "were", "weren't", "what", "whatever", "when", "whence", "whenever",
    "where", "whereabouts", "whereas", "whereby", "whether", "which",
    "while", "whither", "who", "whoever", "whom", "whomsoever",
    "whose", "why", "will", "with", "within", "without", "won't",
    "would", "wouldn't", "you", "your", "yours", "yourself",
    "yourselves",
})
# fmt: on


# ── Index objects (shared across all models) ──────────────────────────────────

def load_index():
    print("Loading index … ", end="", flush=True)
    t0 = time.perf_counter()
    idx    = g.DiskIndex(INDEX)
    pr     = g.PostingsReader(os.path.join(INDEX, PART))
    ls     = g.LengthsSource(os.path.join(INDEX, "lengths"))
    stats  = ls.stats
    print(f"{stats.total_document_count:,} docs, {stats.collection_length:,} tokens "
          f"({time.perf_counter()-t0:.1f}s)")
    return idx, pr, ls, stats


# ── Query helpers ─────────────────────────────────────────────────────────────

def load_queries(path: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def process_query(text: str, stem_fn: Callable[[str], str]) -> List[str]:
    """Tokenise, lowercase, remove INQUERY stops, apply stemmer."""
    tokens = tokenize_string(text)          # lowercases
    return [stem_fn(t) for t in tokens
            if t and t not in INQUERY_STOPS]


def resolve(results: List[Tuple[int, float]], idx) -> List[Tuple[str, float]]:
    return [(idx.get_name(d), s) for d, s in results]


def to_ranked_docs(results: List[Tuple[str, float]]) -> List[RankedDoc]:
    return [RankedDoc(name, score, i) for i, (name, score) in enumerate(results, 1)]


# ── QL (Dirichlet-smoothed Query Likelihood) — C++ DAAT ──────────────────────

def ql_search(stemmed_terms: List[str], idx, ls,
              n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    """Delegate to the C++ DAAT QL implementation for full speed."""
    results = g.ql_search(idx, ls, list(dict.fromkeys(stemmed_terms)),
                           mu=mu, n=n, part=PART)
    return [(sd.document, sd.score) for sd in results]


# ── WSDM-Int (IDF-weighted Dirichlet QL) — C++ DAAT ──────────────────────────

def wsdm_search(stemmed_terms: List[str], idx, pr, ls, N_DOCS: int,
                n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    """IDF-weighted Dirichlet QL via C++ DAAT (WSDM-Int unigram component).

    w_t = log((N+1)/(df_t+0.5)), normalised to sum to 1.
    """
    weighted: List[Tuple[str, float]] = []
    for term in dict.fromkeys(stemmed_terms):
        s = pr.get_stats(term)
        if s is None:
            continue
        idf = math.log((N_DOCS - s["document_count"] + 0.5) / (s["document_count"] + 0.5))
        if idf > 0:
            weighted.append((term, idf))
    if not weighted:
        return []
    results = g.ql_search_weighted(idx, ls, weighted, mu=mu, n=n, part=PART)
    return [(sd.document, sd.score) for sd in results]


# ── SDM (Sequential Dependence Model) ────────────────────────────────────────

_SDM_PREFETCH = 5000  # QL pre-filter depth for 2-stage SDM


def sdm_search(stemmed_terms: List[str], idx, pr, ls, C: int,
               uni_w: float = SDM_UNI, od_w: float = SDM_OD,
               uw_w: float = SDM_UW, win: int = SDM_WIN,
               n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    """Two-stage Sequential Dependence Model.

    Stage 1  — QL pre-filter retrieves _SDM_PREFETCH candidates.
    Stage 2  — positional re-scoring with unigram + ordered (#od:1) +
               unordered (#uw:win) bigram features from read_positions_for.
    Falls back to QL results when fewer than 2 unique terms or when the index
    has no positional data.

    Score = uni_w*QL_uni + od_w*Σ QL_od(t_i,t_{i+1}) + uw_w*Σ QL_uw(t_i,t_{i+1})
    """
    terms = list(dict.fromkeys(stemmed_terms))
    if not terms:
        return []

    if len(terms) < 2:
        return ql_search(stemmed_terms, idx, ls, n=n, mu=mu)

    # ── Stage 1: QL pre-filter ─────────────────────────────────────────────────
    ql_results = ql_search(stemmed_terms, idx, ls, n=_SDM_PREFETCH, mu=mu)
    if not ql_results:
        return []

    candidate_ids = sorted(doc_id for doc_id, _ in ql_results)

    # ── Stage 2: positional data for candidates only ───────────────────────────
    term_stats: List[Tuple[float, Dict[int, List[int]]]] = []
    for term in terms:
        s = pr.get_stats(term)
        p_t = s["collection_count"] / C if s else 0.0
        pos_data = pr.read_positions_for(term, candidate_ids)
        doc_pos  = {doc_id: positions for doc_id, positions in pos_data}
        term_stats.append((p_t, doc_pos))

    if not any(bool(dp) for _, dp in term_stats):
        return ql_results[:n]   # index has no positional data

    # ── Stage 3: SDM re-scoring of candidates ─────────────────────────────────
    scored: List[Tuple[int, float]] = []
    for doc_id in candidate_ids:
        dl    = ls.length(doc_id)
        denom = dl + mu

        # Unigram QL (tf = len(positions))
        uni_score = sum(
            math.log(max(len(dp.get(doc_id, [])) + mu * p_t, 1e-300) / denom)
            for p_t, dp in term_stats
        )

        od_score = 0.0
        uw_score = 0.0
        for i in range(len(terms) - 1):
            _, dp1 = term_stats[i]
            _, dp2 = term_stats[i + 1]
            pos1 = dp1.get(doc_id, [])
            pos2 = dp2.get(doc_id, [])

            # #od:1 — term2 immediately follows term1
            set2  = set(pos2)
            od_tf = sum(1 for p in pos1 if (p + 1) in set2)

            # #uw:win — both terms within win positions (bisect for O(n log m))
            uw_tf = 0
            for p in pos1:
                lo = bisect.bisect_left(pos2,  p - win + 1)
                hi = bisect.bisect_right(pos2, p + win - 1)
                if lo < hi:
                    uw_tf += 1

            cf_bigram = max(od_tf, 1) / C
            od_score += math.log((od_tf + mu * cf_bigram) / denom)
            uw_score += math.log((uw_tf + mu * cf_bigram) / denom)

        scored.append((doc_id, uni_w * uni_score + od_w * od_score + uw_w * uw_score))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── RM3 vocabulary pre-scan ───────────────────────────────────────────────────

def build_rm3_vocab(index_path: str, part: str, pr, C: int,
                    min_df: int = 10, max_df: int = 100_000,
                    sample_size: int = RM3_VOCAB,
                    seed: int = 42) -> List[Tuple[str, float]]:
    """Sample content terms from the index for RM3 expansion."""
    print(f"Building RM3 vocab (min_df={min_df}, max_df={max_df}) … ",
          end="", flush=True)
    t0 = time.perf_counter()

    _reader = g.BTreeReader(os.path.join(index_path, part))
    bt = _reader.iterator()

    candidates: List[Tuple[str, float]] = []
    while not bt.is_done:
        term = bt.key
        s    = pr.get_stats(term)
        if s and min_df <= s["document_count"] <= max_df:
            candidates.append((term, s["collection_count"] / C))
        bt.next_key()

    rng = random.Random(seed)
    rng.shuffle(candidates)
    vocab = candidates[:sample_size]
    print(f"{len(candidates):,} content terms → sampled {len(vocab):,} "
          f"({time.perf_counter()-t0:.1f}s)")
    return vocab


# ── Weighted QL (for RM3 re-query) ───────────────────────────────────────────

def weighted_ql_search(term_weights: List[Tuple[str, float]], pr, ls, C: int,
                       n: int = N, mu: float = MU,
                       cache_doc_ids=None,
                       cache_tfs=None) -> List[Tuple[int, float]]:
    """Weighted Dirichlet QL.  Accepts optional pre-cached posting arrays to
    avoid pybind11 overhead for rm3_vocab terms (see build_rm3_cache)."""
    valid: List[Tuple[float, float, object, object]] = []  # (w, p_t, ids, tfs)
    for term, w in term_weights:
        s = pr.get_stats(term)
        if s is None:
            continue
        p_t = s["collection_count"] / C
        if cache_doc_ids is not None and term in cache_doc_ids:
            valid.append((w, p_t, cache_doc_ids[term], cache_tfs[term]))
        else:
            it = pr.get_postings(term)
            if it is None:
                continue
            valid.append((w, p_t, it, None))   # None → iterate it directly

    if not valid:
        return []

    total_w   = math.fsum(w for w, _, _, _ in valid)
    log_mu_pt = [math.log(mu * p_t) for _, p_t, _, _ in valid]
    sum_w_lmp = math.fsum(w * lmp for (w, _, _, _), lmp in zip(valid, log_mu_pt))

    corrections: Dict[int, float] = defaultdict(float)
    for (w, p_t, ids, tfs), lmp in zip(valid, log_mu_pt):
        if tfs is not None:
            # Cached arrays — Python zip, no C++ round-trip
            for doc_id, tf in zip(ids, tfs):
                corrections[doc_id] += w * (math.log(tf + mu * p_t) - lmp)
        else:
            # Live PostingsIterator
            for doc_id, tf in ids:
                corrections[doc_id] += w * (math.log(tf + mu * p_t) - lmp)

    scored: List[Tuple[int, float]] = []
    for doc_id, corr in corrections.items():
        scored.append((doc_id,
                       sum_w_lmp + corr - total_w * math.log(ls.length(doc_id) + mu)))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── RM3 vocab cache ───────────────────────────────────────────────────────────

def build_rm3_cache(rm3_vocab: List[Tuple[str, float]], pr):
    """Pre-load the fixed rm3_vocab posting lists into compact Python arrays.

    Called once before the query loop.  Per-query RM3 then uses bisect
    instead of skip_to, cutting pybind11 round-trips from ~5 000/query to ~5.

    Returns (cache_doc_ids, cache_tfs): dicts mapping term → array.
    """
    print("Building RM3 posting cache … ", end="", flush=True)
    t0 = time.perf_counter()

    cache_doc_ids: Dict[str, object] = {}
    cache_tfs:     Dict[str, object] = {}

    for term, _ in rm3_vocab:
        it = pr.get_postings(term)
        if it is None:
            continue
        dids: List[int] = []
        tfs:  List[int] = []
        for doc_id, tf in it:
            dids.append(doc_id)
            tfs.append(tf)
        if dids:
            cache_doc_ids[term] = _array('q', dids)
            cache_tfs[term]     = _array('i', tfs)

    print(f"{len(cache_doc_ids):,} terms cached ({time.perf_counter()-t0:.1f}s)")
    return cache_doc_ids, cache_tfs


# ── RM3 ───────────────────────────────────────────────────────────────────────

def rm3_search(stemmed_terms: List[str],
               rm3_vocab: List[Tuple[str, float]],
               idx, pr, ls, C: int,
               n: int = N, mu: float = MU,
               fb_docs: int = FB_DOCS,
               fb_terms: int = FB_TERMS,
               lam: float = RM3_LAM,
               cache_doc_ids=None,
               cache_tfs=None) -> List[Tuple[int, float]]:
    """RM3: initial QL → relevance model → interpolated re-query.

    When cache_doc_ids/cache_tfs (from build_rm3_cache) are provided, TF
    lookup for the fixed rm3_vocab terms uses bisect on pre-loaded arrays
    instead of skip_to, which is ~100x faster per query.
    """
    initial = ql_search(stemmed_terms, idx, ls, n=fb_docs, mu=mu)
    if not initial:
        return []

    max_s  = max(s for _, s in initial)
    exp_s  = [(d, math.exp(s - max_s)) for d, s in initial]
    Z      = sum(w for _, w in exp_s)
    fb_w   = {d: w / Z for d, w in exp_s}
    fb_ids = sorted(fb_w)

    query_vocab: List[Tuple[str, float]] = []
    for term in dict.fromkeys(stemmed_terms):
        s = pr.get_stats(term)
        if s:
            query_vocab.append((term, s["collection_count"] / C))

    all_vocab = {t: p for t, p in query_vocab}
    for t, p in rm3_vocab:
        all_vocab.setdefault(t, p)

    fb_dl = {fb_id: ls.length(fb_id) for fb_id in fb_ids}

    p_t_R: Dict[str, float] = {}
    for term, p_t in all_vocab.items():
        if cache_doc_ids is not None and term in cache_doc_ids:
            # Cached path: bisect lookup, no C++ round-trip per feedback doc
            dids = cache_doc_ids[term]
            tfs  = cache_tfs[term]
            contrib = 0.0
            for fb_id in fb_ids:
                idx = bisect.bisect_left(dids, fb_id)
                tf  = tfs[idx] if idx < len(dids) and dids[idx] == fb_id else 0
                contrib += fb_w[fb_id] * (tf + mu * p_t) / (fb_dl[fb_id] + mu)
        else:
            # Uncached path (query terms): use skip_to as before
            it = pr.get_postings(term)
            if it is None:
                continue
            contrib = 0.0
            for fb_id in fb_ids:
                it.skip_to(fb_id)
                tf = it.count if (not it.is_done and it.doc_id == fb_id) else 0
                contrib += fb_w[fb_id] * (tf + mu * p_t) / (fb_dl[fb_id] + mu)

        if contrib > 0.0:
            p_t_R[term] = contrib

    if not p_t_R:
        return ql_search(stemmed_terms, pr, ls, C, n=n, mu=mu)

    q_len = max(len(stemmed_terms), 1)
    p_t_q = {t: 1.0 / q_len for t in stemmed_terms}

    all_terms = set(p_t_R) | set(p_t_q)
    rm3_raw   = {
        t: lam * p_t_q.get(t, 0.0) + (1.0 - lam) * p_t_R.get(t, 0.0)
        for t in all_terms
    }

    top = sorted(rm3_raw.items(), key=lambda x: -x[1])[:fb_terms]
    total_w = sum(w for _, w in top)
    if total_w <= 0:
        return ql_search(stemmed_terms, pr, ls, C, n=n, mu=mu)
    top_norm = [(t, w / total_w) for t, w in top]

    return weighted_ql_search(top_norm, pr, ls, C, n=n, mu=mu,
                               cache_doc_ids=cache_doc_ids,
                               cache_tfs=cache_tfs)


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(queries_type: str = "titles", output_path: Optional[str] = None):
    # Load index
    idx, pr, ls, stats = load_index()
    C      = stats.collection_length
    N_DOCS = stats.total_document_count

    queries_file = TITLES_TSV if queries_type == "titles" else DESCS_TSV
    queries = load_queries(queries_file)
    qrels   = read_qrels(QRELS_FILE)
    stem    = get_stemmer(STEMMER_NAME)
    print(f"Query set: {queries_type} ({len(queries)} queries)  "
          f"|  Judged topics: {len(qrels)}")
    print(f"Index part: {PART}  Stemmer: {STEMMER_NAME}")

    rm3_vocab = build_rm3_vocab(INDEX, PART, pr, C)
    rm3_cache_doc_ids, rm3_cache_tfs = build_rm3_cache(rm3_vocab, pr)

    bm25_retrieval = Retrieval(INDEX, b=BM25_B, k=BM25_K, part=PART,
                                stemmer=STEMMER_NAME)

    MODEL_NAMES = ["BM25", "QL", "SDM", "WSDM", "RM3"]
    runs:   Dict[str, Run]   = {m: {} for m in MODEL_NAMES}
    timing: Dict[str, float] = {m: 0.0 for m in MODEL_NAMES}

    total = len(queries)
    print(f"\nRunning {total} queries × {len(MODEL_NAMES)} models …\n")

    for qi, (topic, query_text) in enumerate(sorted(queries.items()), 1):
        stemmed = process_query(query_text, stem)
        if not stemmed:
            # Degenerate query (all stopwords): use unstemmed tokens
            stemmed = [stem(t) for t in tokenize_string(query_text) if t]
        if qi % 50 == 1:
            print(f"  [{qi:3d}/{total}] topic {topic}: {query_text!r}",
                  flush=True)

        # ── BM25 ─────────────────────────────────────────────────────────────
        # Remove INQUERY stops before passing to BM25 (Retrieval handles stemming)
        t0 = time.perf_counter()
        filtered_query = " ".join(
            t for t in tokenize_string(query_text) if t not in INQUERY_STOPS
        ) or query_text
        res_bm25 = bm25_retrieval.search(filtered_query, n=N)
        timing["BM25"] += time.perf_counter() - t0
        runs["BM25"][topic] = to_ranked_docs(res_bm25)

        # ── QL ───────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_ql = resolve(ql_search(stemmed, idx, ls), idx)
        timing["QL"] += time.perf_counter() - t0
        runs["QL"][topic] = to_ranked_docs(res_ql)

        # ── SDM ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_sdm = resolve(sdm_search(stemmed, idx, pr, ls, C), idx)
        timing["SDM"] += time.perf_counter() - t0
        runs["SDM"][topic] = to_ranked_docs(res_sdm)

        # ── WSDM (IDF-weighted QL unigrams) ──────────────────────────────────
        t0 = time.perf_counter()
        res_wsdm = resolve(wsdm_search(stemmed, idx, pr, ls, N_DOCS), idx)
        timing["WSDM"] += time.perf_counter() - t0
        runs["WSDM"][topic] = to_ranked_docs(res_wsdm)

        # ── RM3 ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_rm3 = resolve(rm3_search(stemmed, rm3_vocab, idx, pr, ls, C,
                                     cache_doc_ids=rm3_cache_doc_ids,
                                     cache_tfs=rm3_cache_tfs), idx)
        timing["RM3"] += time.perf_counter() - t0
        runs["RM3"][topic] = to_ranked_docs(res_rm3)

    print("\nEvaluating …", flush=True)

    results: Dict[str, Dict[str, float]] = {}
    for model, run in runs.items():
        ranked = {t: [rd.doc_id for rd in docs] for t, docs in run.items()}
        results[model] = evaluate(ranked, qrels, metrics=METRICS)

    # ── Paper comparison ──────────────────────────────────────────────────────
    PAPER_TARGETS = {
        "titles": {
            "QL":   {"map": 0.252, "ndcg@20": 0.412, "p@20": 0.365},
            "BM25": {"map": 0.254, "ndcg@20": 0.412, "p@20": 0.363},
            "SDM":  {"map": 0.263, "ndcg@20": 0.423, "p@20": 0.375},
            "WSDM": {"map": 0.269, "ndcg@20": 0.432, "p@20": 0.382},
        },
        "descs": {
            "QL":   {"map": 0.244, "ndcg@20": 0.389, "p@20": 0.334},
            "BM25": {"map": 0.237, "ndcg@20": 0.390, "p@20": 0.331},
            "SDM":  {"map": 0.258, "ndcg@20": 0.406, "p@20": 0.349},
            "WSDM": {"map": 0.278, "ndcg@20": 0.428, "p@20": 0.365},
        },
    }

    # ── Build output ──────────────────────────────────────────────────────────
    lines: List[str] = []
    lines.append(f"# Robust04 Retrieval Experiment — PyGalago ({queries_type})\n")
    lines.append(f"**Collection:** Robust04  "
                 f"({N_DOCS:,} documents, {C:,} tokens)\n")
    lines.append(f"**Topics:** {len(queries)} TREC {queries_type} queries\n")
    lines.append(f"**Index part:** `{PART}` (Porter2 stemming, INQUERY stops)\n\n")

    lines.append("## Results vs Paper (Table 7, Huston & Croft 2014)\n")
    targets = PAPER_TARGETS.get(queries_type, {})
    metric_labels = {"map": "MAP", "ndcg@20": "NDCG@20", "p@20": "P@20"}
    header  = "| Model | " + " | ".join(f"Ours {l}" for l in metric_labels.values())
    header += " | " + " | ".join(f"Paper {l}" for l in metric_labels.values()) + " |"
    divider = "|-------|" + "|".join(["--------"] * (len(metric_labels) * 2)) + "|"
    lines.append(header)
    lines.append(divider)

    for model_key, model_display in [("BM25","BM25"),("QL","QL"),
                                      ("SDM","SDM"),("WSDM","WSDM-Int")]:
        s = results.get(model_key, {})
        paper = targets.get(model_key, {})
        row = f"| {model_display:<8} |"
        for m in metric_labels:
            row += f" {s.get(m, 0):.4f} |"
        for m in metric_labels:
            row += f" {paper.get(m, 0):.3f} |"
        lines.append(row)

    # RM3 row (no paper target)
    s = results.get("RM3", {})
    row = f"| RM3      |"
    for m in metric_labels:
        row += f" {s.get(m, 0):.4f} |"
    row += " — | — | — |"
    lines.append(row)

    lines.append("")
    lines.append("## Timing\n")
    lines.append("| Model | Total (s) | Per query (ms) |")
    lines.append("|-------|-----------|----------------|")
    for model in MODEL_NAMES:
        t_total = timing[model]
        t_per   = t_total / total * 1000
        lines.append(f"| {model:<5} | {t_total:7.1f} | {t_per:>14.0f} |")

    markdown = "\n".join(lines)
    print("\n" + markdown)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown + "\n")
        print(f"\nResults written to {output_path!r}")

    return results, timing


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robust04 retrieval experiment")
    ap.add_argument("--queries", choices=["titles", "descs"], default="titles",
                    help="Which query set to use (default: titles)")
    ap.add_argument("--output", default=None,
                    help="Output markdown file")
    args = ap.parse_args()
    if args.output is None:
        args.output = f"results/robust04_{args.queries}.md"
    run_experiment(args.queries, args.output)
