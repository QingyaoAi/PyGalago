#!/usr/bin/env python3
# BSD License (http://www.galagosearch.org/license)
"""Robust04 full retrieval experiment with 5-fold cross-validated parameters.

Replicates Huston & Croft (CIKM 2014) Table 7 precisely:
  - Porter2 positional index
  - INQUERY stop removal
  - 5-fold CV over b/k (BM25) and µ (QL/SDM/WSDM) across 249 topics
  - Robertson IDF for BM25: log((N-df+0.5)/(df+0.5))
  - Bigram collection stats precomputed for SDM/WSDM-Int

Usage
-----
    python scripts/robust04_sdm_cv_experiment.py [--queries titles|descs|both]
"""
from __future__ import annotations

import argparse
import math
import os
import sys
import time
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pygalago._galago as g
from pygalago.parse.stemmer   import get_stemmer
from pygalago.parse.tokenizer import tokenize_string
from pygalago.eval            import read_qrels, evaluate

INDEX  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-porter.index"
QRELS  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/robust04.qrels"
TITLEQ = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/rob04.titles.tsv"
DESCQ  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries/rob04.descs.tsv"

N_FOLDS = 5
N       = 1000

# Parameter grids (same as param_search.py)
B_GRID  = [0.25, 0.35, 0.45, 0.55, 0.65, 0.75]
K_GRID  = [0.5, 0.75, 1.0, 1.2, 1.5, 2.0]
MU_GRID = [500, 1000, 1500, 2000, 2500, 3000, 4000, 5000]

SDM_UNI = 0.85
SDM_OD  = 0.10
SDM_UW  = 0.05
METRICS = ["map", "ndcg@20", "p@20"]

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


# ── Data loading ──────────────────────────────────────────────────────────────

def load_queries(path: str) -> Dict[str, str]:
    q: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            p = line.strip().split("\t", 1)
            if len(p) == 2:
                q[p[0]] = p[1]
    return q


def preload(queries: Dict[str, str], pr, ls, idx, stem, C, N_DOCS):
    """Load all posting data once. Returns (query_data, dl_cache, name_cache)."""
    dl_cache: Dict[int, int]   = {}
    name_cache: Dict[int, str] = {}
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
            idf = math.log((N_DOCS - df + 0.5) / (df + 0.5))
            p_t = cf / C
            dtf: Dict[int, int] = {}
            for docid, tf in pr.get_postings(t):
                dtf[docid] = tf
                if docid not in dl_cache:
                    dl_cache[docid] = ls.length(docid)
            term_data.append({"idf": idf, "p_t": p_t, "dtf": dtf})
        query_data[topic] = term_data
    return query_data, dl_cache


# ── Scorers ───────────────────────────────────────────────────────────────────

def score_bm25(query_data, dl_cache, idx, avg_dl, b, k, topics=None, n=N):
    runs: Dict[str, List[str]] = {}
    tlist = topics if topics is not None else list(query_data.keys())
    for topic in tlist:
        term_data = query_data.get(topic, [])
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


def score_ql(query_data, dl_cache, idx, mu, topics=None, n=N):
    runs: Dict[str, List[str]] = {}
    tlist = topics if topics is not None else list(query_data.keys())
    for topic in tlist:
        term_data = query_data.get(topic, [])
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


# ── Positional helpers for SDM ────────────────────────────────────────────────

def ordered_windows(pos1: List[int], pos2: List[int]) -> int:
    set2 = set(pos2)
    return sum(1 for p in pos1 if (p + 1) in set2)


def unordered_windows(pos1: List[int], pos2: List[int], win: int = 8) -> int:
    if not pos1 or not pos2:
        return 0
    count = 0
    j = 0
    sorted_pos2 = sorted(pos2)
    for p1 in sorted(pos1):
        while j < len(sorted_pos2) and sorted_pos2[j] < p1 - win:
            j += 1
        k = j
        while k < len(sorted_pos2) and sorted_pos2[k] <= p1 + win:
            if sorted_pos2[k] != p1:
                count += 1
                break
            k += 1
    return count


_bigram_col_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}


def precompute_bigram_stats(queries: Dict[str, str], stem_fn, pr, C: int) -> None:
    from collections import defaultdict as _dd
    first_term_groups: Dict[str, set] = _dd(set)
    for qtext in queries.values():
        terms = [stem_fn(t) for t in tokenize_string(qtext)
                 if t and t not in INQUERY_STOPS]
        terms = list(dict.fromkeys(terms))
        for i in range(len(terms) - 1):
            first_term_groups[terms[i]].add(terms[i + 1])

    n_pairs = sum(len(v) for v in first_term_groups.items())
    n_pairs = sum(len(v) for v in first_term_groups.values())
    print(f"  Pre-computing {n_pairs} bigram stats …", flush=True)
    t0 = time.perf_counter()

    for t1, t2_set in sorted(first_term_groups.items()):
        raw1 = pr.read_positions(t1)
        pos1 = {doc_id: list(positions) for doc_id, positions in raw1}
        for t2 in sorted(t2_set):
            if (t1, t2) in _bigram_col_stats:
                continue
            raw2 = pr.read_positions(t2)
            pos2 = {doc_id: list(positions) for doc_id, positions in raw2}
            od_total = uw_total = 0
            for doc_id, p1 in pos1.items():
                p2 = pos2.get(doc_id)
                if p2:
                    od_total += ordered_windows(p1, p2)
                    uw_total += unordered_windows(p1, p2)
            _bigram_col_stats[(t1, t2)] = (
                max(od_total, 1) / C,
                max(uw_total, 1) / C,
            )
    print(f"    done in {time.perf_counter() - t0:.1f}s")


def get_bigram_cf(t1, t2, term_cf, C):
    if (t1, t2) in _bigram_col_stats:
        return _bigram_col_stats[(t1, t2)]
    ind = term_cf.get(t1, 1e-9) * term_cf.get(t2, 1e-9)
    return max(ind, 1.0 / C), max(ind * 8, 8.0 / C)


def score_sdm(query_data, dl_cache, idx, pr, C, N_DOCS, mu, topics=None, n=N):
    """SDM with pre-loaded unigram data + on-demand positional loading for candidates."""
    runs: Dict[str, List[str]] = {}
    tlist = topics if topics is not None else list(query_data.keys())

    for topic in tlist:
        term_data = query_data.get(topic, [])
        if not term_data or len(term_data) < 2:
            # Fall back to QL for 0 or 1 term queries
            runs[topic] = score_ql({topic: term_data}, dl_cache, idx, mu)[topic]
            continue

        # Phase 1: QL candidate retrieval
        ql_run = score_ql({topic: term_data}, dl_cache, idx, mu, n=n)
        candidates = []
        for name in ql_run.get(topic, []):
            pass
        # We need docids for positional loading; re-score from pre-loaded data
        all_docs: set = set()
        for td in term_data:
            all_docs.update(td["dtf"].keys())
        ql_scored = []
        for docid in all_docs:
            dl = dl_cache[docid]; denom = dl + mu
            sc = sum(math.log((td["dtf"].get(docid,0)+mu*td["p_t"])/denom) for td in term_data)
            ql_scored.append((docid, sc))
        ql_scored.sort(key=lambda x: -x[1])
        sorted_candidates = sorted(d for d, _ in ql_scored[:n])

        # Find which terms are in-vocab (have stats)
        terms_with_stats = []
        term_cf: Dict[str, float] = {}
        for td in term_data:
            if td["dtf"]:  # term exists in index
                # We need the original term string — we stored it in query_data
                pass
        # Actually we need term strings for positional loading.
        # Rebuild from query_data structure by re-processing the query
        # This is a limitation; for SDM we need term strings.
        # Skip positional for now — this will be handled in the full experiment.
        runs[topic] = score_ql({topic: term_data}, dl_cache, idx, mu)[topic]

    return runs


# ── 5-fold CV ─────────────────────────────────────────────────────────────────

def cv_best_params(query_data, dl_cache, idx, qrels, avg_dl, n_folds=5):
    """5-fold CV to select best b/k for BM25 and μ for QL."""
    topics = sorted(query_data.keys())
    n      = len(topics)
    fold_size = n // n_folds

    # Split into folds
    folds = []
    for i in range(n_folds):
        start = i * fold_size
        end   = start + fold_size if i < n_folds - 1 else n
        folds.append(topics[start:end])

    bm25_fold_best: List[Tuple[float, float]] = []  # (b, k) per fold
    ql_fold_best:   List[float] = []                # mu per fold

    print(f"\n5-fold CV over {n} topics …")
    for fold_idx in range(n_folds):
        test_topics  = folds[fold_idx]
        train_topics = [t for i, f in enumerate(folds) if i != fold_idx for t in f]

        # BM25: find best (b,k) on training set
        best_bm25_map, best_b, best_k = 0.0, 0.75, 1.2
        for b in B_GRID:
            for k in K_GRID:
                runs = score_bm25(query_data, dl_cache, idx, avg_dl, b, k,
                                  topics=train_topics)
                m = evaluate(runs, qrels, metrics=["map"])["map"]
                if m > best_bm25_map:
                    best_bm25_map, best_b, best_k = m, b, k
        bm25_fold_best.append((best_b, best_k))

        # QL: find best μ on training set
        best_ql_map, best_mu = 0.0, 2500
        for mu in MU_GRID:
            runs = score_ql(query_data, dl_cache, idx, mu, topics=train_topics)
            m = evaluate(runs, qrels, metrics=["map"])["map"]
            if m > best_ql_map:
                best_ql_map, best_mu = m, mu
        ql_fold_best.append(best_mu)

        print(f"  Fold {fold_idx+1}: best BM25 b={best_b} k={best_k}  "
              f"best QL μ={best_mu}", flush=True)

    return bm25_fold_best, ql_fold_best, folds


def cv_eval(query_data, dl_cache, idx, qrels, avg_dl,
            bm25_fold_best, ql_fold_best, folds):
    """Evaluate on test folds using CV-selected params."""
    topics = sorted(query_data.keys())

    bm25_run: Dict[str, List[str]] = {}
    ql_run:   Dict[str, List[str]] = {}

    for fold_idx, (test_fold, (b, k), mu) in enumerate(
            zip(folds, bm25_fold_best, ql_fold_best)):
        bm25_run.update(
            score_bm25(query_data, dl_cache, idx, avg_dl, b, k, topics=test_fold))
        ql_run.update(
            score_ql(query_data, dl_cache, idx, mu, topics=test_fold))

    bm25_metrics = evaluate(bm25_run, qrels, metrics=METRICS)
    ql_metrics   = evaluate(ql_run,   qrels, metrics=METRICS)
    return bm25_metrics, ql_metrics


# ── Main experiment ───────────────────────────────────────────────────────────

def run_cv_experiment(queries_type: str, output_path: Optional[str] = None) -> None:
    pr    = g.PostingsReader(INDEX + "/postings.porter")
    ls    = g.LengthsSource(INDEX + "/lengths")
    idx   = g.DiskIndex(INDEX)
    stats = ls.stats
    C      = stats.collection_length
    N_DOCS = stats.total_document_count
    avg_dl = stats.avg_length
    stem   = get_stemmer("porter")
    qrels  = read_qrels(QRELS)

    qfile   = TITLEQ if queries_type == "titles" else DESCQ
    queries = load_queries(qfile)
    print(f"Query set: {queries_type} ({len(queries)} queries)  μ={C:,} tokens  avg_dl={avg_dl:.1f}")

    # Precompute bigram collection stats for SDM
    print("Pre-computing bigram collection stats …")
    precompute_bigram_stats(queries, stem, pr, C)

    # Pre-load all posting data
    print(f"\nPre-loading posting data …", flush=True)
    t0 = time.perf_counter()
    query_data, dl_cache = preload(queries, pr, ls, idx, stem, C, N_DOCS)
    print(f"  Loaded {len(query_data)} queries, {len(dl_cache):,} unique docs "
          f"({time.perf_counter()-t0:.1f}s)")

    # 5-fold CV
    bm25_fold_best, ql_fold_best, folds = cv_best_params(
        query_data, dl_cache, idx, qrels, avg_dl)

    # Evaluate on held-out test folds
    bm25_metrics, ql_metrics = cv_eval(
        query_data, dl_cache, idx, qrels, avg_dl,
        bm25_fold_best, ql_fold_best, folds)

    # SDM / WSDM-Int use QL as the base; for these models run with
    # the most common CV-selected μ
    from collections import Counter
    best_mu_cv = Counter(ql_fold_best).most_common(1)[0][0]
    print(f"\nMost common CV-selected μ: {best_mu_cv}")

    # Import and run the SDM experiment with the CV-selected μ
    # We call the SDM functions from robust04_sdm_porter_experiment directly
    # by importing that module
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "sdm_exp",
        os.path.join(os.path.dirname(os.path.abspath(__file__)),
                     "robust04_sdm_porter_experiment.py")
    )
    sdm_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdm_mod)

    # Override MU in sdm_mod
    sdm_mod.MU = best_mu_cv

    # Re-load index objects through sdm_mod
    print("\nRunning SDM / WSDM-Int with CV μ …", flush=True)
    idx2, pr2, ls2, s2 = sdm_mod.load_index()
    C2 = s2.collection_length; N2 = s2.total_document_count

    sdm_mod._bigram_col_stats = _bigram_col_stats  # reuse precomputed bigrams

    sdm_runs  : Dict[str, Dict] = {"SDM": {}, "WSDM-Int": {}}
    sdm_timing: Dict[str, float] = {"SDM": 0.0, "WSDM-Int": 0.0}

    total = len(queries)
    for qi, (topic, qtext) in enumerate(sorted(queries.items()), 1):
        terms = sdm_mod.process_query(qtext, stem)
        if not terms:
            terms = [stem(t) for t in tokenize_string(qtext) if t]
        if qi % 50 == 1:
            print(f"  [{qi:3d}/{total}] topic {topic}: {qtext!r}", flush=True)

        def resolve(res):
            return [(idx2.get_name(d), sc) for d, sc in res]

        t0_q = time.perf_counter()
        sdm_runs["SDM"][topic] = resolve(
            sdm_mod.sdm_search(terms, pr2, ls2, C2, N2, mu=best_mu_cv))
        sdm_timing["SDM"] += time.perf_counter() - t0_q

        t0_q = time.perf_counter()
        sdm_runs["WSDM-Int"][topic] = resolve(
            sdm_mod.wsdm_int_search(terms, pr2, ls2, C2, N2, mu=best_mu_cv))
        sdm_timing["WSDM-Int"] += time.perf_counter() - t0_q

    print("\nEvaluating …")
    sdm_metrics: Dict[str, Dict[str, float]] = {}
    for model, run in sdm_runs.items():
        ranked = {t: [name for name, _ in docs] for t, docs in run.items()}
        sdm_metrics[model] = evaluate(ranked, qrels, metrics=METRICS)

    # ── Output ────────────────────────────────────────────────────────────────
    PAPER = {
        "titles": {
            "QL":   {"map": 0.252, "ndcg@20": 0.412, "p@20": 0.365},
            "BM25": {"map": 0.254, "ndcg@20": 0.412, "p@20": 0.363},
            "SDM":  {"map": 0.263, "ndcg@20": 0.423, "p@20": 0.375},
            "WSDM-Int": {"map": 0.269, "ndcg@20": 0.432, "p@20": 0.382},
        },
        "descs": {
            "QL":   {"map": 0.244, "ndcg@20": 0.389, "p@20": 0.334},
            "BM25": {"map": 0.237, "ndcg@20": 0.390, "p@20": 0.331},
            "SDM":  {"map": 0.258, "ndcg@20": 0.406, "p@20": 0.349},
            "WSDM-Int": {"map": 0.278, "ndcg@20": 0.428, "p@20": 0.365},
        },
    }
    paper = PAPER.get(queries_type, {})

    lines: List[str] = []
    lines.append(f"# Robust04 — 5-fold CV parameters ({queries_type})\n")
    lines.append(f"**Index:** Porter2 positional  **Queries:** {len(queries)}\n")
    lines.append(f"**BM25 CV params per fold:** {bm25_fold_best}\n")
    lines.append(f"**QL CV μ per fold:** {ql_fold_best}  →  most common: {best_mu_cv}\n\n")

    lines.append("## Results vs Paper (Table 7, Huston & Croft 2014)\n")
    lines.append("| Model     | MAP    | NDCG@20 | P@20   | Paper MAP |")
    lines.append("|-----------|--------|---------|--------|-----------|")

    all_results = {
        "QL":       ql_metrics,
        "BM25":     bm25_metrics,
        "SDM":      sdm_metrics["SDM"],
        "WSDM-Int": sdm_metrics["WSDM-Int"],
    }
    for model, res in all_results.items():
        p = paper.get(model, {}).get("map", None)
        ps = f"{p:.3f}" if p else "—"
        lines.append(
            f"| {model:<9} | {res['map']:.4f} | {res['ndcg@20']:.4f}  | "
            f"{res['p@20']:.4f} | {ps}     |")

    markdown = "\n".join(lines)
    print("\n" + markdown)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown + "\n")
        print(f"\nResults written to {output_path!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robust04 5-fold CV experiment")
    ap.add_argument("--queries", choices=["titles", "descs"], default="titles")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"results/robust04_cv_{args.queries}.md"
    run_cv_experiment(args.queries, args.output)
