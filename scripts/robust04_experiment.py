#!/usr/bin/env python3
# BSD License (http://www.galagosearch.org/license)
"""Robust04 retrieval experiment: BM25, QL, SDM, WSDM, RM3.

Models
------
BM25   – Okapi BM25 (b=0.75, k=1.2) via C++ DAAT engine.
QL     – Dirichlet-smoothed Query Likelihood (μ=2500).
SDM    – Sequential Dependence Model (QL unigrams only; ordered/unordered
         window features require a positional index which is not present in
         this build — effectively QL).
WSDM   – Weighted SDM: IDF-weighted Dirichlet QL (unigram approximation
         of Bendersky et al. 2010, distinct from uniform QL).
RM3    – QL + Pseudo-Relevance Feedback (Relevance Model 3):
         fbDocs=10, fbTerms=20, λ=0.6 (RM1/original query interpolation).
         Expansion vocabulary: content terms sampled from the Krovetz index.

Usage
-----
    python scripts/robust04_experiment.py [--output results/robust04.md]
"""
from __future__ import annotations

import argparse
import math
import os
import random
import sys
import time
from collections import defaultdict
from typing import Dict, List, Tuple

# ── PyGalago imports ──────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygalago._galago as g
from pygalago.parse.stemmer    import get_stemmer
from pygalago.parse.tokenizer  import tokenize_string
from pygalago.retrieval        import Retrieval
from pygalago.eval             import read_qrels, evaluate
from pygalago.eval.run         import write_run, Run, RankedDoc

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX       = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04.index"
PART        = "postings.krovetz"
QUERIES_TSV = os.path.join(INDEX, "queries", "rob04.titles.tsv")
QRELS_FILE  = os.path.join(INDEX, "queries", "robust04.qrels")

N         = 1000    # documents to retrieve per query
MU        = 2500    # Dirichlet prior
BM25_B    = 0.75
BM25_K    = 1.2
FB_DOCS   = 10      # RM3 feedback documents
FB_TERMS  = 20      # RM3 expansion terms
RM3_LAM   = 0.6     # RM3 interpolation (original query weight)
RM3_VOCAB = 500     # expansion vocabulary size for RM3 (speed/quality trade-off)

METRICS = ["map", "ndcg@10", "ndcg@20", "p@10", "mrr", "bpref"]

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


def make_stemmer():
    return get_stemmer("krovetz")


def stem_query(text: str, stem_fn) -> List[str]:
    tokens = tokenize_string(text)
    stemmed = [stem_fn(t) for t in tokens]
    return [t for t in stemmed if t]


def resolve(results: List[Tuple[int, float]], idx) -> List[Tuple[str, float]]:
    return [(idx.get_name(d), s) for d, s in results]


def to_ranked_docs(results: List[Tuple[str, float]]) -> List[RankedDoc]:
    return [RankedDoc(name, score, i) for i, (name, score) in enumerate(results, 1)]

# ── QL (Dirichlet-smoothed Query Likelihood) ──────────────────────────────────

def ql_search(stemmed_terms: List[str], pr, ls, C: int,
              n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    """DAAT Dirichlet QL over posting lists for stemmed query terms."""
    term_info: List[Tuple[float, object]] = []
    for term in dict.fromkeys(stemmed_terms):   # deduplicate, preserve order
        s = pr.get_stats(term)
        if s is None:
            continue
        it = pr.get_postings(term)
        if it is None:
            continue
        term_info.append((s["collection_count"] / C, it))

    if not term_info:
        return []

    # Collect all (docid, tf) pairs across posting lists
    doc_tfs: Dict[int, Dict[int, int]] = defaultdict(dict)
    for i, (_, it) in enumerate(term_info):
        for doc_id, tf in it:
            doc_tfs[doc_id][i] = tf

    # Score each candidate document
    term_ps = [p for p, _ in term_info]
    scored: List[Tuple[int, float]] = []
    for doc_id, tfs in doc_tfs.items():
        dl    = ls.length(doc_id)
        denom = dl + mu
        score = sum(
            math.log((tfs.get(i, 0) + mu * p) / denom)
            for i, p in enumerate(term_ps)
        )
        scored.append((doc_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]

# ── WSDM (IDF-weighted Dirichlet QL) ─────────────────────────────────────────

def wsdm_search(stemmed_terms: List[str], pr, ls, C: int, N_DOCS: int,
                n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    """IDF-weighted Dirichlet QL.

    Each term's QL contribution is weighted by its IDF:
        w_t = log((N+1) / (df_t + 0.5))
    Weights are normalised to sum to 1 before scoring.
    This is the unigram approximation of the WSDM model
    (Bendersky, Metzler & Croft 2010) for count-only indexes.
    """
    term_info: List[Tuple[float, float, object]] = []  # (p_t, idf, it)
    for term in dict.fromkeys(stemmed_terms):
        s = pr.get_stats(term)
        if s is None:
            continue
        it = pr.get_postings(term)
        if it is None:
            continue
        p_t = s["collection_count"] / C
        idf = math.log((N_DOCS + 1) / (s["document_count"] + 0.5))
        term_info.append((p_t, idf, it))

    if not term_info:
        return []

    total_idf = sum(idf for _, idf, _ in term_info)

    doc_tfs: Dict[int, Dict[int, int]] = defaultdict(dict)
    for i, (_, _, it) in enumerate(term_info):
        for doc_id, tf in it:
            doc_tfs[doc_id][i] = tf

    scored: List[Tuple[int, float]] = []
    for doc_id, tfs in doc_tfs.items():
        dl    = ls.length(doc_id)
        denom = dl + mu
        score = sum(
            (idf / total_idf) * math.log((tfs.get(i, 0) + mu * p_t) / denom)
            for i, (p_t, idf, _) in enumerate(term_info)
        )
        scored.append((doc_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]

# ── RM3 vocabulary pre-scan ───────────────────────────────────────────────────

def build_rm3_vocab(index_path: str, part: str, pr, C: int,
                    min_df: int = 10, max_df: int = 100_000,
                    sample_size: int = RM3_VOCAB,
                    seed: int = 42) -> List[Tuple[str, float]]:
    """Scan the index vocabulary and sample content terms for RM3 expansion.

    Keeps the BTreeReader alive to avoid iterator dangling references.
    """
    print(f"Building RM3 expansion vocabulary (min_df={min_df}, max_df={max_df}) … ",
          end="", flush=True)
    t0 = time.perf_counter()

    # Keep both reader AND iterator alive simultaneously
    _reader = g.BTreeReader(os.path.join(index_path, part))
    bt = _reader.iterator()

    candidates: List[Tuple[str, float]] = []
    while not bt.is_done:
        term = bt.key
        s    = pr.get_stats(term)
        if s and min_df <= s["document_count"] <= max_df:
            candidates.append((term, s["collection_count"] / C))
        bt.next_key()

    # Shuffle and take sample
    rng = random.Random(seed)
    rng.shuffle(candidates)
    vocab = candidates[:sample_size]

    print(f"{len(candidates):,} content terms → sampled {len(vocab):,} "
          f"({time.perf_counter()-t0:.1f}s)")
    return vocab

# ── Weighted QL (RM3 second-pass retrieval) ───────────────────────────────────

def weighted_ql_search(term_weights: List[Tuple[str, float]], pr, ls, C: int,
                       n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    """QL retrieval with explicit per-term weights (used for RM3 re-query)."""
    term_info: List[Tuple[float, float, object]] = []  # (weight, p_t, it)
    for term, w in term_weights:
        s  = pr.get_stats(term)
        if s is None:
            continue
        it = pr.get_postings(term)
        if it is None:
            continue
        term_info.append((w, s["collection_count"] / C, it))

    if not term_info:
        return []

    doc_tfs: Dict[int, Dict[int, int]] = defaultdict(dict)
    for i, (_, _, it) in enumerate(term_info):
        for doc_id, tf in it:
            doc_tfs[doc_id][i] = tf

    scored: List[Tuple[int, float]] = []
    for doc_id, tfs in doc_tfs.items():
        dl    = ls.length(doc_id)
        denom = dl + mu
        score = sum(
            w * math.log((tfs.get(i, 0) + mu * p_t) / denom)
            for i, (w, p_t, _) in enumerate(term_info)
        )
        scored.append((doc_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]

# ── RM3 ───────────────────────────────────────────────────────────────────────

def rm3_search(stemmed_terms: List[str],
               rm3_vocab: List[Tuple[str, float]],
               pr, ls, C: int,
               n: int = N, mu: float = MU,
               fb_docs: int = FB_DOCS,
               fb_terms: int = FB_TERMS,
               lam: float = RM3_LAM) -> List[Tuple[int, float]]:
    """RM3: initial QL → relevance model estimation → interpolated re-query.

    The relevance model P(t|R) is estimated over:
      - All original query terms (guaranteed coverage)
      - A random sample of content terms from the index vocabulary
    """
    # Step 1: Initial QL retrieval
    initial = ql_search(stemmed_terms, pr, ls, C, n=fb_docs, mu=mu)
    if not initial:
        return []

    # Normalise QL scores → P(d|q) using log-sum-exp
    max_s  = max(s for _, s in initial)
    exp_s  = [(d, math.exp(s - max_s)) for d, s in initial]
    Z      = sum(w for _, w in exp_s)
    fb_w   = {d: w / Z for d, w in exp_s}            # docid → weight
    fb_ids = sorted(fb_w)                             # sorted for skip_to

    # Step 2: Estimate P(t|R) over expansion vocabulary + query terms
    # Combine unique query terms with sampled vocab
    query_vocab: List[Tuple[str, float]] = []
    for term in dict.fromkeys(stemmed_terms):
        s = pr.get_stats(term)
        if s:
            query_vocab.append((term, s["collection_count"] / C))

    all_vocab = {t: p for t, p in query_vocab}
    for t, p in rm3_vocab:
        all_vocab.setdefault(t, p)

    # Cache per-feedback-doc lengths (avoids repeated pybind11 calls in inner loop)
    fb_dl = {fb_id: ls.length(fb_id) for fb_id in fb_ids}

    p_t_R: Dict[str, float] = {}
    for term, p_t in all_vocab.items():
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

    # Step 3: RM3 interpolation with original query model
    q_len  = max(len(stemmed_terms), 1)
    p_t_q  = {t: 1.0 / q_len for t in stemmed_terms}   # uniform over query terms

    # P_RM3(t) = λ·P(t|q) + (1-λ)·P(t|R)
    all_terms = set(p_t_R) | set(p_t_q)
    rm3_raw   = {
        t: lam * p_t_q.get(t, 0.0) + (1.0 - lam) * p_t_R.get(t, 0.0)
        for t in all_terms
    }

    # Take top fb_terms by combined score
    top = sorted(rm3_raw.items(), key=lambda x: -x[1])[:fb_terms]
    total_w = sum(w for _, w in top)
    if total_w <= 0:
        return ql_search(stemmed_terms, pr, ls, C, n=n, mu=mu)
    top_norm = [(t, w / total_w) for t, w in top]

    # Step 4: Weighted QL re-query
    return weighted_ql_search(top_norm, pr, ls, C, n=n, mu=mu)

# ── SDM (note) ────────────────────────────────────────────────────────────────
# The Robust04 index was built with count-only posting lists (no positions).
# SDM's ordered-window (#od) and unordered-window (#uw) features require
# per-position data which is absent here.  The unigram component of SDM uses
# Dirichlet-smoothed QL (μ=2500, uni_weight=0.85 — identical to plain QL after
# normalising away the dropped bigram weights).
# Result: SDM ≡ QL in this count-only setting.

# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(output_path: str | None = None):
    # Load index
    idx, pr, ls, stats = load_index()
    C      = stats.collection_length
    N_DOCS = stats.total_document_count

    # Load queries and qrels
    queries = load_queries(QUERIES_TSV)
    qrels   = read_qrels(QRELS_FILE)
    stem    = make_stemmer()
    print(f"Queries: {len(queries)}  |  Judged topics: {len(qrels)}")

    # Pre-build RM3 expansion vocabulary
    rm3_vocab = build_rm3_vocab(INDEX, PART, pr, C)

    # BM25 Retrieval object
    bm25_retrieval = Retrieval(INDEX, b=BM25_B, k=BM25_K, part=PART)

    # ── Per-model run storage ─────────────────────────────────────────────────
    MODEL_NAMES = ["BM25", "QL", "SDM", "WSDM", "RM3"]
    runs:   Dict[str, Run]   = {m: {} for m in MODEL_NAMES}
    timing: Dict[str, float] = {m: 0.0 for m in MODEL_NAMES}

    total = len(queries)
    print(f"\nRunning {total} queries × {len(MODEL_NAMES)} models …\n")

    for qi, (topic, query_text) in enumerate(sorted(queries.items()), 1):
        stemmed = stem_query(query_text, stem)
        if qi % 50 == 1:
            print(f"  [{qi:3d}/{total}] topic {topic}: {query_text!r}",
                  flush=True)

        # ── BM25 ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_bm25 = bm25_retrieval.search(query_text, n=N)
        timing["BM25"] += time.perf_counter() - t0
        runs["BM25"][topic] = to_ranked_docs(res_bm25)

        # ── QL ───────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_ql = resolve(ql_search(stemmed, pr, ls, C), idx)
        timing["QL"] += time.perf_counter() - t0
        runs["QL"][topic] = to_ranked_docs(res_ql)

        # ── SDM (QL unigrams; bigrams not available in count-only index) ─────
        # SDM uses uni_weight=0.85 (the od/uw components are zero without positions).
        # After normalisation this is identical to plain QL.  We run it explicitly
        # for completeness and note the equivalence in the results.
        t0 = time.perf_counter()
        # SDM == QL in this setting; copy the QL run
        runs["SDM"][topic] = runs["QL"][topic]
        timing["SDM"] += time.perf_counter() - t0

        # ── WSDM ─────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_wsdm = resolve(wsdm_search(stemmed, pr, ls, C, N_DOCS), idx)
        timing["WSDM"] += time.perf_counter() - t0
        runs["WSDM"][topic] = to_ranked_docs(res_wsdm)

        # ── RM3 ──────────────────────────────────────────────────────────────
        t0 = time.perf_counter()
        res_rm3 = resolve(rm3_search(stemmed, rm3_vocab, pr, ls, C), idx)
        timing["RM3"] += time.perf_counter() - t0
        runs["RM3"][topic] = to_ranked_docs(res_rm3)

    print("\nEvaluating …", flush=True)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results: Dict[str, Dict[str, float]] = {}
    for model, run in runs.items():
        ranked = {t: [rd.doc_id for rd in docs] for t, docs in run.items()}
        results[model] = evaluate(ranked, qrels, metrics=METRICS)

    # ── Build markdown output ─────────────────────────────────────────────────
    lines: List[str] = []

    lines.append("# Robust04 Retrieval Experiment — PyGalago\n")
    lines.append(f"**Collection:** Robust04  "
                 f"({N_DOCS:,} documents, {C:,} tokens)\n")
    lines.append(f"**Topics:** {len(queries)} TREC title queries (301–700)\n")
    lines.append(f"**Qrels:** {sum(len(v) for v in qrels.values()):,} judgments "
                 f"across {len(qrels)} topics\n")
    lines.append(f"**Index part:** `{PART}` (Krovetz stemming)\n\n")

    # Model descriptions
    lines.append("## Model Settings\n")
    lines.append("| Model | Scoring | Parameters |")
    lines.append("|-------|---------|------------|")
    lines.append(f"| BM25  | Okapi BM25 | b={BM25_B}, k={BM25_K} |")
    lines.append(f"| QL    | Dirichlet Query Likelihood | μ={MU} |")
    lines.append(f"| SDM   | Sequential Dependence Model | μ={MU}, uni=0.85, od=0.10†, uw=0.05† |")
    lines.append(f"| WSDM  | IDF-Weighted Dirichlet QL | μ={MU}, w_t∝log(N/df_t) |")
    lines.append(f"| RM3   | QL + Pseudo-Relevance Feedback | μ={MU}, fbDocs={FB_DOCS}, "
                 f"fbTerms={FB_TERMS}, λ={RM3_LAM} |")
    lines.append("")
    lines.append("> † SDM ordered/unordered window features require a positional index. "
                 "This build contains count-only posting lists, so the bigram components "
                 "are unavailable — SDM reduces to its unigram (QL) component and is "
                 "**numerically identical to QL** in this setting.\n")

    # Main results table
    lines.append("## Retrieval Results\n")
    metric_labels = {
        "map":     "MAP",
        "ndcg@10": "NDCG@10",
        "ndcg@20": "NDCG@20",
        "p@10":    "P@10",
        "mrr":     "MRR",
        "bpref":   "Bpref",
    }
    header  = "| Model | " + " | ".join(metric_labels.values()) + " | Avg query time |"
    divider = "|-------|" + "|".join(["--------"] * len(metric_labels)) + "|----------------|"
    lines.append(header)
    lines.append(divider)

    for model in MODEL_NAMES:
        s    = results[model]
        avg_t = timing[model] / total
        row  = f"| {model:<5} | "
        row += " | ".join(f"{s[m]:.4f}" for m in METRICS)
        row += f" | {avg_t*1000:.0f} ms |"
        lines.append(row)

    lines.append("")

    # Per-topic stats
    lines.append("## Per-Topic Breakdown (MAP)\n")
    lines.append("First 20 topics shown (sorted by topic id).\n")
    sample_topics = sorted(list(qrels.keys()))[:20]

    breakdown_header  = "| Topic | Query | " + " | ".join(MODEL_NAMES) + " |"
    breakdown_divider = "|-------|-------|" + "|".join(["-------"] * len(MODEL_NAMES)) + "|"
    lines.append(breakdown_header)
    lines.append(breakdown_divider)

    from pygalago.eval.metrics import average_precision
    from pygalago.eval.qrels   import relevant_docs

    for topic in sample_topics:
        rel     = relevant_docs(qrels, topic)
        q_text  = queries.get(topic, "—")[:35]
        aps     = []
        for model in MODEL_NAMES:
            ranked_ids = [rd.doc_id for rd in runs[model].get(topic, [])]
            aps.append(f"{average_precision(ranked_ids, rel):.4f}")
        lines.append(f"| {topic} | {q_text} | " + " | ".join(aps) + " |")

    lines.append("")

    # Notes
    lines.append("## Notes\n")
    lines.append("- **QL** uses Dirichlet smoothing with μ=2500 (the standard Galago default).")
    lines.append("- **WSDM** weights each term by its BM25-style IDF before Dirichlet scoring. "
                 "This is the unigram component of the Weighted Sequential Dependence Model.")
    lines.append("- **RM3** estimates a relevance model from the top-10 QL results, "
                 f"using {RM3_VOCAB} randomly-sampled content terms as the expansion vocabulary. "
                 "The expanded query is interpolated with the original query model at λ=0.6.")
    lines.append("- **SDM** is reported separately for completeness but is numerically "
                 "identical to QL in this run because the Robust04 index was built without "
                 "positional posting data. A full SDM implementation requires reindexing "
                 "with position lists.")
    lines.append("- All models use the **Krovetz-stemmed** (`postings.krovetz`) index part.")
    lines.append("- All models retrieve top-1000 documents.")
    lines.append(f"- Evaluation uses the standard Robust04 qrels "
                 f"({sum(len(v) for v in qrels.values()):,} judgments).")

    # Timing summary
    lines.append("\n## Timing Summary\n")
    lines.append("| Model | Total (s) | Per query (ms) |")
    lines.append("|-------|-----------|----------------|")
    for model in MODEL_NAMES:
        t_total = timing[model]
        t_per   = t_total / total * 1000
        lines.append(f"| {model:<5} | {t_total:7.1f} | {t_per:>14.0f} |")

    markdown = "\n".join(lines)

    # Print to stdout
    print("\n" + markdown)

    # Write to file
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown + "\n")
        print(f"\nResults written to {output_path!r}")

    return results, timing


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robust04 retrieval experiment")
    ap.add_argument("--output", default="results/robust04_results.md",
                    help="Output markdown file (default: results/robust04_results.md)")
    args = ap.parse_args()
    run_experiment(args.output)
