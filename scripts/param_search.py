#!/usr/bin/env python3
"""Fast parameter grid search for BM25 (b, k) and QL (mu) on Robust04.

Pre-loads all posting data once, then sweeps parameters with pure arithmetic.
Reports best params for titles and descs separately, then re-runs the full
experiment with the best title params (matching paper's approach).
"""
from __future__ import annotations

import math
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygalago._galago as g
from pygalago.parse.stemmer   import get_stemmer
from pygalago.parse.tokenizer import tokenize_string
from pygalago.eval            import read_qrels, evaluate

INDEX  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-porter.index"
QRELS  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/robust04.qrels"
TITLEQ = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/rob04.titles.tsv"
DESCQ  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/rob04.descs.tsv"

# fmt: off
INQUERY_STOPS = frozenset({
    "a","about","above","according","across","after","afterwards","again","against",
    "albeit","all","almost","alone","along","already","also","although","always","am",
    "among","amongst","an","and","another","any","are","around","as","at","be","became",
    "because","been","before","being","below","beside","besides","between","beyond",
    "both","but","by","can","could","did","do","does","doing","done","down","during",
    "each","either","else","enough","even","ever","every","except","few","for","from",
    "further","get","given","go","got","had","has","have","having","he","her","here",
    "him","his","how","however","i","if","in","into","is","it","its","just","less",
    "like","many","may","me","might","more","most","much","my","no","nobody","none",
    "not","now","nowhere","of","off","on","or","other","others","our","out","over",
    "per","rather","same","since","so","some","still","such","than","that","the",
    "their","them","then","there","these","they","this","those","though","through",
    "thus","till","to","too","us","very","was","we","were","what","when","where",
    "which","while","who","whom","whose","why","will","with","would","you","your",
})
# fmt: on

B_VALUES  = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85]
K_VALUES  = [0.5, 0.75, 1.0, 1.2, 1.5, 2.0]
MU_VALUES = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 7500]


def load_queries(path: str) -> Dict[str, str]:
    q: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split("\t", 1)
            if len(p) == 2:
                q[p[0]] = p[1]
    return q


def preload(queries: Dict[str, str], pr, ls, idx, stem, C, N_DOCS):
    """Pre-load all posting data once. Returns (query_data, dl_cache)."""
    dl_cache: Dict[int, int] = {}
    query_data: Dict[str, list] = {}

    for topic, qtext in sorted(queries.items()):
        terms = list(dict.fromkeys(
            stem(t) for t in tokenize_string(qtext)
            if t and t not in INQUERY_STOPS
        ))
        term_data = []
        for t in terms:
            s = pr.get_stats(t)
            if not s:
                continue
            df = s["document_count"]
            cf = s["collection_count"]
            idf = math.log((N_DOCS - df + 0.5) / (df + 0.5))  # Robertson
            p_t = cf / C
            dtf: Dict[int, int] = {}
            for docid, tf in pr.get_postings(t):
                dtf[docid] = tf
                if docid not in dl_cache:
                    dl_cache[docid] = ls.length(docid)
            term_data.append({"idf": idf, "p_t": p_t, "dtf": dtf})
        query_data[topic] = term_data
    return query_data, dl_cache


def score_bm25(query_data, dl_cache, idx, avg_dl, b, k, n=1000):
    runs: Dict[str, List[str]] = {}
    for topic, term_data in query_data.items():
        if not term_data:
            runs[topic] = []
            continue
        all_docs: set = set()
        for td in term_data:
            all_docs.update(td["dtf"].keys())
        scored: List[Tuple[int, float]] = []
        for docid in all_docs:
            dl   = dl_cache[docid]
            norm = 1.0 - b + b * dl / avg_dl
            sc   = sum(
                td["idf"] * td["dtf"].get(docid, 0) * (k + 1)
                / (td["dtf"].get(docid, 0) + k * norm)
                for td in term_data
            )
            scored.append((docid, sc))
        scored.sort(key=lambda x: -x[1])
        runs[topic] = [idx.get_name(d) for d, _ in scored[:n]]
    return runs


def score_ql(query_data, dl_cache, idx, mu, n=1000):
    runs: Dict[str, List[str]] = {}
    for topic, term_data in query_data.items():
        if not term_data:
            runs[topic] = []
            continue
        all_docs: set = set()
        for td in term_data:
            all_docs.update(td["dtf"].keys())
        scored: List[Tuple[int, float]] = []
        for docid in all_docs:
            dl    = dl_cache[docid]
            denom = dl + mu
            sc    = sum(
                math.log((td["dtf"].get(docid, 0) + mu * td["p_t"]) / denom)
                for td in term_data
            )
            scored.append((docid, sc))
        scored.sort(key=lambda x: -x[1])
        runs[topic] = [idx.get_name(d) for d, _ in scored[:n]]
    return runs


def grid_search(label: str, query_data, dl_cache, idx, qrels, avg_dl):
    print(f"\n=== BM25 grid — {label} ===")
    best_bm25 = (0.0, 0.75, 1.2)
    for b in B_VALUES:
        for k in K_VALUES:
            runs = score_bm25(query_data, dl_cache, idx, avg_dl, b, k)
            m = evaluate(runs, qrels, metrics=["map"])["map"]
            if m > best_bm25[0]:
                best_bm25 = (m, b, k)
            print(f"  b={b:.2f}  k={k:.2f}  MAP={m:.4f}", flush=True)
    print(f"Best BM25 {label}: MAP={best_bm25[0]:.4f}  b={best_bm25[1]}  k={best_bm25[2]}")

    print(f"\n=== QL mu grid — {label} ===")
    best_ql = (0.0, 2500)
    for mu in MU_VALUES:
        runs = score_ql(query_data, dl_cache, idx, mu)
        m = evaluate(runs, qrels, metrics=["map"])["map"]
        if m > best_ql[0]:
            best_ql = (m, mu)
        print(f"  mu={mu:5d}  MAP={m:.4f}", flush=True)
    print(f"Best QL {label}: MAP={best_ql[0]:.4f}  mu={best_ql[1]}")

    return best_bm25, best_ql


def main():
    pr    = g.PostingsReader(INDEX + "/postings.porter")
    ls    = g.LengthsSource(INDEX + "/lengths")
    idx   = g.DiskIndex(INDEX)
    stats = ls.stats
    C      = stats.collection_length
    N_DOCS = stats.total_document_count
    avg_dl = stats.avg_length
    stem   = get_stemmer("porter")
    qrels  = read_qrels(QRELS)

    print(f"Index: {N_DOCS:,} docs, {C:,} tokens, avg_dl={avg_dl:.1f}")

    # ── Titles ────────────────────────────────────────────────────────────────
    print("\nPre-loading title postings …", flush=True)
    title_q = load_queries(TITLEQ)
    title_data, dl_cache = preload(title_q, pr, ls, idx, stem, C, N_DOCS)
    print(f"  Loaded {len(title_data)} title queries, {len(dl_cache):,} unique docs")
    best_bm25_t, best_ql_t = grid_search(
        "titles", title_data, dl_cache, idx, qrels, avg_dl)

    # ── Descs ─────────────────────────────────────────────────────────────────
    print("\nPre-loading desc postings …", flush=True)
    desc_q = load_queries(DESCQ)
    # Desc queries touch more docs; reuse dl_cache and extend it
    desc_data, dl_cache = preload(desc_q, pr, ls, idx, stem, C, N_DOCS)
    print(f"  Loaded {len(desc_data)} desc queries, {len(dl_cache):,} unique docs")
    best_bm25_d, best_ql_d = grid_search(
        "descs", desc_data, dl_cache, idx, qrels, avg_dl)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("OPTIMAL PARAMETERS SUMMARY")
    print("="*60)
    print(f"{'Model':<12} {'Qset':<8} {'Param':<20} {'MAP':>8}")
    print("-"*55)
    print(f"{'BM25':<12} {'titles':<8} b={best_bm25_t[1]:.2f}, k={best_bm25_t[2]:.2f}  "
          f"{'MAP':>8}={best_bm25_t[0]:.4f}  (paper: 0.254)")
    print(f"{'BM25':<12} {'descs':<8} b={best_bm25_d[1]:.2f}, k={best_bm25_d[2]:.2f}  "
          f"{'MAP':>8}={best_bm25_d[0]:.4f}  (paper: 0.237)")
    print(f"{'QL':<12} {'titles':<8} mu={best_ql_t[1]:<16}  "
          f"{'MAP':>8}={best_ql_t[0]:.4f}  (paper: 0.252)")
    print(f"{'QL':<12} {'descs':<8} mu={best_ql_d[1]:<16}  "
          f"{'MAP':>8}={best_ql_d[0]:.4f}  (paper: 0.244)")


if __name__ == "__main__":
    main()
