#!/usr/bin/env python3
# BSD License (http://www.galagosearch.org/license)
"""Robust04 SDM and WSDM-Int experiment using the Porter2-stemmed positional index.

Uses robust04-porter.index which was built by build_robust04_porter.py with
positional posting lists and Porter2 (Snowball English) stemming.

Paper targets (Huston & Croft 2014, Table 7):
  Title:  QL=0.252, BM25=0.254, SDM=0.263, WSDM-Int=0.269
  Desc:   QL=0.244, BM25=0.237, SDM=0.258, WSDM-Int=0.278

Usage
-----
    python scripts/robust04_sdm_porter_experiment.py [--queries titles|descs]
    python scripts/robust04_sdm_porter_experiment.py --queries descs --output results/robust04_sdm_porter_descs.md
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

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-porter.index"
PART   = "postings.porter"
# Queries and qrels live in the complete-index queries directory
_Q_DIR = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index/queries"
QRELS   = os.path.join(_Q_DIR, "robust04.qrels")
TITLE_Q = os.path.join(_Q_DIR, "rob04.titles.tsv")
DESC_Q  = os.path.join(_Q_DIR, "rob04.descs.tsv")

N      = 1000
MU     = 2500
BM25_B = 0.75
BM25_K = 1.2
SDM_UNI = 0.85
SDM_OD  = 0.10
SDM_UW  = 0.05
METRICS = ["map", "ndcg@20", "p@20"]

# INQUERY stop words (same 418-word set as robust04_experiment.py)
# fmt: off
INQUERY_STOPS = frozenset({
    "a","about","above","according","across","after","afterwards","again",
    "against","albeit","all","almost","alone","along","already","also",
    "although","always","am","among","amongst","an","and","another","any",
    "anybody","anyhow","anyone","anything","anyway","anywhere","apart","are",
    "around","as","at","av","be","became","because","become","becomes",
    "becoming","been","before","beforehand","behind","being","below","beside",
    "besides","between","beyond","both","but","by","can","cannot","canst",
    "certain","cf","choose","contrariwise","cos","could","couldn't","dare",
    "daren't","definitely","despite","did","didn't","different","directly",
    "do","does","doesn't","doing","done","don't","down","during","e","each",
    "eg","either","else","elsewhere","enough","etc","even","ever","every",
    "everybody","everyone","everything","everywhere","except","exactly","far",
    "few","ff","fifth","first","following","for","former","formerly","forth",
    "from","further","furthermore","get","given","go","got","h","had",
    "hadn't","has","hasn't","have","haven't","having","he","her","here",
    "hereabouts","hereafter","hereby","herein","hereinafter","heretofore",
    "hereunder","hereupon","herewith","him","himself","his","how","however",
    "i","ie","if","in","indeed","inside","instead","into","is","isn't","it",
    "its","itself","just","kind","kg","km","last","latter","latterly","less",
    "lest","let","like","little","lots","many","may","maybe","me","meantime",
    "meanwhile","might","moreover","most","mostly","more","mr","mrs","much",
    "my","myself","namely","needn't","neither","never","nevertheless","next",
    "no","nobody","none","noone","nothing","notwithstanding","now","nowhere",
    "of","off","often","ok","on","once","one","only","onto","or","other",
    "others","otherwise","ought","our","ours","ourselves","out","outside",
    "over","own","per","perhaps","please","rather","re","really","regarding",
    "same","sans","self","several","should","shouldn't","since","so","some",
    "somebody","somehow","someone","something","sometime","sometimes",
    "somewhere","still","such","than","that","the","thee","their","theirs",
    "them","themselves","then","thence","there","thereabouts","thereafter",
    "thereby","therfore","therefore","therein","these","they","this","those",
    "thou","though","through","throughout","thru","thus","thy","till","to",
    "together","too","toward","towards","under","unless","until","up","upon",
    "us","very","via","vs","was","wasn't","we","were","weren't","what",
    "whatever","when","whence","whenever","where","whereabouts","whereas",
    "whereby","whether","which","while","whither","who","whoever","whom",
    "whomsoever","whose","why","will","with","within","without","won't",
    "would","wouldn't","you","your","yours","yourself","yourselves",
})
# fmt: on


# ── Index loading ─────────────────────────────────────────────────────────────

def load_index():
    print("Loading index … ", end="", flush=True)
    t0 = time.perf_counter()
    idx   = g.DiskIndex(INDEX)
    pr    = g.PostingsReader(os.path.join(INDEX, PART))
    ls    = g.LengthsSource(os.path.join(INDEX, "lengths"))
    stats = ls.stats
    print(f"{stats.total_document_count:,} docs, {stats.collection_length:,} tokens "
          f"({time.perf_counter()-t0:.1f}s)")
    return idx, pr, ls, stats


def load_queries(path: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def process_query(text: str, stem_fn: Callable[[str], str]) -> List[str]:
    """Tokenise, lowercase, remove INQUERY stops, apply Porter2 stemmer."""
    tokens = tokenize_string(text)
    return [stem_fn(t) for t in tokens if t and t not in INQUERY_STOPS]


# ── QL retrieval ──────────────────────────────────────────────────────────────

def ql_search(terms: List[str], pr, ls, C: int,
              n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    term_info: List[Tuple[float, object]] = []
    for t in dict.fromkeys(terms):
        s = pr.get_stats(t)
        if s is None:
            continue
        it = pr.get_postings(t)
        if it is None:
            continue
        term_info.append((s["collection_count"] / C, it))
    if not term_info:
        return []
    doc_tfs: Dict[int, Dict[int, int]] = defaultdict(dict)
    for i, (_, it) in enumerate(term_info):
        for doc_id, tf in it:
            doc_tfs[doc_id][i] = tf
    term_ps = [p for p, _ in term_info]
    scored = []
    for doc_id, tfs in doc_tfs.items():
        dl    = ls.length(doc_id)
        denom = dl + mu
        score = sum(math.log((tfs.get(i, 0) + mu * p) / denom)
                    for i, p in enumerate(term_ps))
        scored.append((doc_id, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── BM25 ──────────────────────────────────────────────────────────────────────

def bm25_search(terms: List[str], pr, ls, C: int, N_DOCS: int,
                n: int = N, b: float = BM25_B, k: float = BM25_K
                ) -> List[Tuple[int, float]]:
    avg_dl = ls.stats.avg_length
    term_info = []
    for t in dict.fromkeys(terms):
        s = pr.get_stats(t)
        if s is None:
            continue
        it = pr.get_postings(t)
        if it is None:
            continue
        df  = s["document_count"]
        idf = math.log((N_DOCS - df + 0.5) / (df + 0.5))  # Robertson IDF (Galago Java)
        term_info.append((idf, it))
    if not term_info:
        return []
    doc_tfs: Dict[int, Dict[int, int]] = defaultdict(dict)
    for i, (_, it) in enumerate(term_info):
        for doc_id, tf in it:
            doc_tfs[doc_id][i] = tf
    scored = []
    for doc_id, tfs in doc_tfs.items():
        dl = ls.length(doc_id)
        norm = 1.0 - b + b * dl / avg_dl
        score = sum(
            idf * (tf * (k + 1)) / (tf + k * norm)
            for i, (idf, _) in enumerate(term_info)
            for tf in [tfs.get(i, 0)]
        )
        scored.append((doc_id, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── Positional helpers ────────────────────────────────────────────────────────

def load_doc_positions(pr, term: str,
                       candidates: Optional[List[int]] = None) -> Dict[int, List[int]]:
    if candidates is not None:
        raw = pr.read_positions_for(term, candidates)
    else:
        raw = pr.read_positions(term)
    return {doc_id: list(positions) for doc_id, positions in raw}


_bigram_col_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}


def precompute_bigram_stats(queries: Dict[str, str],
                            stem_fn: Callable[[str], str],
                            pr, C: int) -> None:
    """Pre-scan full positional posting lists to get actual od/uw collection counts."""
    from collections import defaultdict as _dd
    first_term_groups: Dict[str, set] = _dd(set)
    for qtext in queries.values():
        terms = process_query(qtext, stem_fn)
        for i in range(len(terms) - 1):
            first_term_groups[terms[i]].add(terms[i + 1])

    n_pairs = sum(len(v) for v in first_term_groups.values())
    print(f"  Pre-computing {n_pairs} bigram collection stats …", flush=True)
    t0 = time.perf_counter()

    for t1, t2_set in sorted(first_term_groups.items()):
        pos1 = load_doc_positions(pr, t1)
        for t2 in sorted(t2_set):
            if (t1, t2) in _bigram_col_stats:
                continue
            pos2 = load_doc_positions(pr, t2)
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


def get_bigram_cf(t1: str, t2: str,
                  term_cf: Dict[str, float], C: int) -> Tuple[float, float]:
    if (t1, t2) in _bigram_col_stats:
        return _bigram_col_stats[(t1, t2)]
    ind = term_cf.get(t1, 1e-9) * term_cf.get(t2, 1e-9)
    return max(ind, 1.0 / C), max(ind * 8, 8.0 / C)


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


# ── SDM (Sequential Dependence Model) ────────────────────────────────────────

def sdm_search(terms: List[str], pr, ls, C: int, N_DOCS: int,
               n: int = N, mu: float = MU,
               uni_w: float = SDM_UNI,
               od_w: float  = SDM_OD,
               uw_w: float  = SDM_UW) -> List[Tuple[int, float]]:
    terms = list(dict.fromkeys(terms))
    if not terms:
        return []

    # Phase 1: QL candidate retrieval (fast, count-only)
    candidates_scored = ql_search(terms, pr, ls, C, n=n, mu=mu)
    if not candidates_scored:
        return []
    candidate_set    = {doc_id for doc_id, _ in candidates_scored}
    sorted_candidates = sorted(candidate_set)

    # Phase 2: Load positions for candidates (in-vocab terms only)
    term_positions: Dict[str, Dict[int, List[int]]] = {}
    term_cf: Dict[str, float] = {}
    vocab_terms: List[str] = []
    for t in terms:
        s = pr.get_stats(t)
        if s is None:
            continue
        vocab_terms.append(t)
        term_positions[t] = load_doc_positions(pr, t, sorted_candidates)
        term_cf[t] = max(s["collection_count"] / C, 1e-9)
    if not vocab_terms:
        return ql_search(terms, pr, ls, C, n=n, mu=mu)

    # Phase 3: Score candidates with full SDM
    scored: List[Tuple[int, float]] = []
    for doc_id in candidate_set:
        dl    = ls.length(doc_id)
        denom = dl + mu

        uni_score = sum(
            math.log((len(term_positions[t].get(doc_id, [])) + mu * term_cf[t]) / denom)
            for t in vocab_terms
        )

        od_score = 0.0
        uw_score = 0.0
        for i in range(len(vocab_terms) - 1):
            t1, t2 = vocab_terms[i], vocab_terms[i + 1]
            pos1 = term_positions[t1].get(doc_id, [])
            pos2 = term_positions[t2].get(doc_id, [])
            od_tf = ordered_windows(pos1, pos2)
            uw_tf = unordered_windows(pos1, pos2)
            od_bg, uw_bg = get_bigram_cf(t1, t2, term_cf, C)
            od_score += math.log((od_tf + mu * od_bg) / denom)
            uw_score += math.log((uw_tf + mu * uw_bg) / denom)

        total = uni_w * uni_score + od_w * od_score + uw_w * uw_score
        scored.append((doc_id, total))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── WSDM-Int ──────────────────────────────────────────────────────────────────

def wsdm_int_search(terms: List[str], pr, ls, C: int, N_DOCS: int,
                    n: int = N, mu: float = MU,
                    uni_w: float = SDM_UNI,
                    od_w: float  = SDM_OD,
                    uw_w: float  = SDM_UW) -> List[Tuple[int, float]]:
    terms = list(dict.fromkeys(terms))
    if not terms:
        return []

    candidates_scored = ql_search(terms, pr, ls, C, n=n, mu=mu)
    if not candidates_scored:
        return []
    candidate_set     = {doc_id for doc_id, _ in candidates_scored}
    sorted_candidates = sorted(candidate_set)

    term_positions: Dict[str, Dict[int, List[int]]] = {}
    term_cf: Dict[str, float] = {}
    term_df: Dict[str, int]   = {}
    vocab_terms: List[str] = []
    for t in terms:
        s = pr.get_stats(t)
        if s is None:
            continue
        vocab_terms.append(t)
        term_positions[t] = load_doc_positions(pr, t, sorted_candidates)
        term_cf[t] = max(s["collection_count"] / C, 1e-9)
        term_df[t] = s["document_count"]
    if not vocab_terms:
        return ql_search(terms, pr, ls, C, n=n, mu=mu)

    def idf_fn(df: int) -> float:
        return math.log((N_DOCS + 1) / (df + 0.5)) if df > 0 else 0.0

    uni_idfs  = {t: idf_fn(term_df[t]) for t in vocab_terms}
    total_idf = sum(uni_idfs.values())
    if total_idf <= 0:
        return ql_search(terms, pr, ls, C, n=n, mu=mu)
    n_bigrams = max(len(vocab_terms) - 1, 1)

    bigram_od: Dict = {}
    bigram_uw: Dict = {}
    for i in range(len(vocab_terms) - 1):
        t1, t2 = vocab_terms[i], vocab_terms[i + 1]
        key = (t1, t2)
        for d in sorted_candidates:
            p1 = term_positions[t1].get(d, [])
            p2 = term_positions[t2].get(d, [])
            bigram_od[(key, d)] = ordered_windows(p1, p2)
            bigram_uw[(key, d)] = unordered_windows(p1, p2)

    scored: List[Tuple[int, float]] = []
    for doc_id in candidate_set:
        dl    = ls.length(doc_id)
        denom = dl + mu
        score = 0.0
        for t in vocab_terms:
            tf = len(term_positions[t].get(doc_id, []))
            score += (uni_w * uni_idfs[t] / total_idf) * math.log(
                (tf + mu * term_cf[t]) / denom)
        for i in range(len(vocab_terms) - 1):
            t1, t2 = vocab_terms[i], vocab_terms[i + 1]
            key = (t1, t2)
            od_tf = bigram_od.get((key, doc_id), 0)
            uw_tf = bigram_uw.get((key, doc_id), 0)
            od_bg, uw_bg = get_bigram_cf(t1, t2, term_cf, C)
            score += (od_w / n_bigrams) * math.log((od_tf + mu * od_bg) / denom)
            score += (uw_w / n_bigrams) * math.log((uw_tf + mu * uw_bg) / denom)
        scored.append((doc_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── WSDM unigram-only ─────────────────────────────────────────────────────────

def wsdm_unigram_search(terms: List[str], pr, ls, C: int, N_DOCS: int,
                        n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    term_info = []
    for t in dict.fromkeys(terms):
        s = pr.get_stats(t)
        if s is None:
            continue
        it = pr.get_postings(t)
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
    scored = []
    for doc_id, tfs in doc_tfs.items():
        dl    = ls.length(doc_id)
        denom = dl + mu
        score = sum(
            (idf / total_idf) * math.log((tfs.get(i, 0) + mu * p_t) / denom)
            for i, (p_t, idf, _) in enumerate(term_info))
        scored.append((doc_id, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(queries_type: str = "titles",
                   output_path: Optional[str] = None,
                   mu: float = MU,
                   bm25_b: float = BM25_B,
                   bm25_k: float = BM25_K) -> None:
    idx, pr, ls, stats = load_index()
    C      = stats.collection_length
    N_DOCS = stats.total_document_count

    qfile   = TITLE_Q if queries_type == "titles" else DESC_Q
    queries = load_queries(qfile)
    qrels   = read_qrels(QRELS)
    stem    = get_stemmer("porter")

    print(f"Query set: {queries_type} ({len(queries)} queries)  "
          f"|  Judged topics: {len(qrels)}")
    print(f"Index: {PART}  (Porter2, positional)  μ={mu}  b={bm25_b}  k={bm25_k}")

    precompute_bigram_stats(queries, stem, pr, C)

    MODEL_NAMES = ["QL", "BM25", "SDM", "WSDM-Uni", "WSDM-Int"]
    runs:   Dict[str, Dict] = {m: {} for m in MODEL_NAMES}
    timing: Dict[str, float] = {m: 0.0 for m in MODEL_NAMES}

    total = len(queries)
    print(f"\nRunning {total} queries × {len(MODEL_NAMES)} models …\n")

    for qi, (topic, query_text) in enumerate(sorted(queries.items()), 1):
        terms = process_query(query_text, stem)
        if not terms:
            terms = [stem(t) for t in tokenize_string(query_text) if t]
        if qi % 50 == 1:
            print(f"  [{qi:3d}/{total}] topic {topic}: {query_text!r}", flush=True)

        def resolve(res):
            return [(idx.get_name(d), s) for d, s in res]

        t0 = time.perf_counter()
        runs["QL"][topic] = resolve(ql_search(terms, pr, ls, C, mu=mu))
        timing["QL"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        runs["BM25"][topic] = resolve(bm25_search(terms, pr, ls, C, N_DOCS, b=bm25_b, k=bm25_k))
        timing["BM25"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        runs["SDM"][topic] = resolve(sdm_search(terms, pr, ls, C, N_DOCS, mu=mu))
        timing["SDM"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        runs["WSDM-Uni"][topic] = resolve(
            wsdm_unigram_search(terms, pr, ls, C, N_DOCS, mu=mu))
        timing["WSDM-Uni"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        runs["WSDM-Int"][topic] = resolve(
            wsdm_int_search(terms, pr, ls, C, N_DOCS, mu=mu))
        timing["WSDM-Int"] += time.perf_counter() - t0

    print("\nEvaluating …", flush=True)
    results: Dict[str, Dict[str, float]] = {}
    for model, run in runs.items():
        ranked = {t: [name for name, _ in docs] for t, docs in run.items()}
        results[model] = evaluate(ranked, qrels, metrics=METRICS)

    PAPER = {
        "titles": {
            "QL": {"map": 0.252, "ndcg@20": 0.412, "p@20": 0.365},
            "BM25": {"map": 0.254, "ndcg@20": 0.412, "p@20": 0.363},
            "SDM": {"map": 0.263, "ndcg@20": 0.423, "p@20": 0.375},
            "WSDM-Int": {"map": 0.269, "ndcg@20": 0.432, "p@20": 0.382},
        },
        "descs": {
            "QL": {"map": 0.244, "ndcg@20": 0.389, "p@20": 0.334},
            "BM25": {"map": 0.237, "ndcg@20": 0.390, "p@20": 0.331},
            "SDM": {"map": 0.258, "ndcg@20": 0.406, "p@20": 0.349},
            "WSDM-Int": {"map": 0.278, "ndcg@20": 0.428, "p@20": 0.365},
        },
    }
    paper = PAPER.get(queries_type, {})

    lines = [f"# Robust04 SDM/WSDM-Int — Porter2 positional index ({queries_type})\n"]
    lines.append(f"**Index:** `{PART}` (Porter2, positional)  "
                 f"**Queries:** {len(queries)}  **μ:** {MU}\n")
    lines.append(f"**SDM weights:** uni={SDM_UNI}, od={SDM_OD}, uw={SDM_UW}\n\n")

    lines.append("## Results vs Paper (Huston & Croft 2014, Table 7)\n")
    hdr = "| Model     | MAP    | NDCG@20 | P@20   | Paper MAP |"
    div = "|-----------|--------|---------|--------|-----------|"
    lines.append(hdr)
    lines.append(div)
    paper_map = {
        "QL": "QL", "BM25": "BM25", "SDM": "SDM",
        "WSDM-Uni": None, "WSDM-Int": "WSDM-Int",
    }
    for m in MODEL_NAMES:
        r  = results[m]
        pk = paper_map.get(m)
        p_map = paper.get(pk, {}).get("map", None) if pk else None
        p_str = f"{p_map:.3f}" if p_map is not None else "—"
        lines.append(
            f"| {m:<9s} | {r['map']:.4f} | {r['ndcg@20']:.4f}  | "
            f"{r['p@20']:.4f} | {p_str}     |"
        )

    lines.append("")
    lines.append("## Timing\n")
    lines.append("| Model     | Total (s) | Per query (ms) |")
    lines.append("|-----------|-----------|----------------|")
    for m in MODEL_NAMES:
        t_total = timing[m]
        t_per   = t_total / total * 1000
        lines.append(f"| {m:<9s} | {t_total:7.1f} | {t_per:>14.0f} |")

    markdown = "\n".join(lines)
    print("\n" + markdown)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(markdown + "\n")
        print(f"\nResults written to {output_path!r}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Robust04 SDM/WSDM-Int experiment — Porter2 positional index")
    ap.add_argument("--queries", choices=["titles", "descs"], default="titles")
    ap.add_argument("--output", default=None)
    ap.add_argument("--mu",  type=float, default=MU,     help="Dirichlet μ (default: %(default)s)")
    ap.add_argument("--b",   type=float, default=BM25_B, help="BM25 b (default: %(default)s)")
    ap.add_argument("--k",   type=float, default=BM25_K, help="BM25 k1 (default: %(default)s)")
    args = ap.parse_args()
    if args.output is None:
        args.output = f"results/robust04_sdm_porter_{args.queries}.md"
    run_experiment(args.queries, args.output, mu=args.mu, bm25_b=args.b, bm25_k=args.k)
