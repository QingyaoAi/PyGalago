#!/usr/bin/env python3
# BSD License (http://www.galagosearch.org/license)
"""Robust04 SDM and WSDM-Int experiment using the complete index with positions.

Uses robust04-complete-index which has positional posting lists, enabling
proper SDM (#od:1 and #uw:8) and WSDM-Int (IDF-weighted unigrams + bigrams).

Paper targets (Huston & Croft 2014, Table 7):
  Title:  QL=0.252, BM25=0.254, SDM=0.263, WSDM-Int=0.269
  Desc:   QL=0.244, BM25=0.237, SDM=0.258, WSDM-Int=0.278

Usage
-----
    python scripts/robust04_sdm_experiment.py [--queries titles|descs]
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
from pygalago.parse.tokenizer import tokenize_string
from pygalago.eval            import read_qrels, evaluate

try:
    from nltk.stem import WordNetLemmatizer as _WNL
    _wn_lemmatizer = _WNL()
    _HAS_WORDNET = True
except Exception:
    _HAS_WORDNET = False

# ── Configuration ─────────────────────────────────────────────────────────────

INDEX  = "/Users/Aqy/Documents/Graduate_study/CIIR/Project/robust04-complete-index"
PART   = "postings.krovetz"   # positional Krovetz-stemmed posting lists
QRELS  = os.path.join(INDEX, "queries", "robust04.qrels")
TITLE_Q = os.path.join(INDEX, "queries", "rob04.titles.tsv")
DESC_Q  = os.path.join(INDEX, "queries", "rob04.descs.tsv")

N      = 1000
MU     = 2500
SDM_UNI = 0.85
SDM_OD  = 0.10
SDM_UW  = 0.05
METRICS = ["map", "ndcg@20", "p@20"]

# INQUERY stop words
INQUERY_STOPS = frozenset({
    "a","about","above","according","across","after","afterwards","again",
    "against","all","almost","alone","along","already","also","although",
    "always","am","among","amongst","an","and","another","any","are",
    "around","as","at","be","because","been","before","being","below",
    "beside","besides","between","beyond","both","but","by","can","could",
    "did","different","do","does","doing","done","down","during","each",
    "either","else","enough","even","ever","every","except","few","for",
    "from","further","get","given","go","got","had","has","have","having",
    "he","her","here","him","his","how","however","i","if","in","into",
    "is","it","its","just","less","like","many","may","me","might","more",
    "most","much","my","no","nobody","none","not","now","nowhere","of",
    "off","on","or","other","others","our","out","over","per","rather",
    "same","since","so","some","still","such","than","that","the","their",
    "them","then","there","these","they","this","those","though","through",
    "thus","till","to","too","us","very","was","we","were","what","when",
    "where","which","while","who","whom","whose","why","will","with",
    "would","you","your",
})


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


_krovetz_cache: Dict[str, Optional[str]] = {}

# Suffix → replacement pairs, most-specific first.
# Applied only when the term is OOV in the Krovetz index.
_STRIP_RULES: List[Tuple[str, str]] = [
    ("izations", "ize"), ("ization",  "ize"),
    ("ication",  "ify"),
    ("nesses",   ""),    ("ness",     ""),
    ("ments",    ""),    ("ment",     ""),
    ("istics",   "ist"), ("istic",    "ist"),
    ("isms",     ""),    ("ism",      ""),
    ("ings",     ""),    ("ing",      ""),
    ("ations",   "ate"), ("ation",    "ate"),
    ("ations",   ""),    ("ation",    ""),
    ("itions",   ""),    ("ition",    ""),
    ("ities",    "e"),   ("ity",      "e"),
    ("ities",    ""),    ("ity",      ""),
    ("icals",    ""),    ("ical",     ""),
    ("ics",      ""),    ("ic",       ""),
    ("ives",     ""),    ("ive",      ""),
    ("ers",      ""),    ("er",       ""),
    ("als",      ""),    ("al",       ""),
    ("ous",      ""),
    ("ors",      ""),    ("or",       ""),
    ("s",        ""),
]


def krovetz_approx(term: str, pr) -> Optional[str]:
    """Best-match query normalization for the Krovetz index.

    Tries in order:
      1. As-is
      2. WordNet noun / verb lemma
      3. Nationality adjective → country noun (-an, -ian, -ish, -ese, -ine)
      4. Suffix stripping (approximate Krovetz rules for OOV terms)
    Returns a form with df > 0, or None if nothing works.
    """
    if term in _krovetz_cache:
        return _krovetz_cache[term]

    def _try(t: str) -> bool:
        return bool(t) and pr.get_stats(t) is not None

    # 1. As-is
    if _try(term):
        _krovetz_cache[term] = term
        return term

    # 2. WordNet noun/verb lemma
    if _HAS_WORDNET:
        for pos in ("n", "v"):
            lemma = _wn_lemmatizer.lemmatize(term, pos=pos)
            if lemma != term and _try(lemma):
                _krovetz_cache[term] = lemma
                return lemma

    # 3. Nationality-adjective → country-noun heuristics
    if len(term) > 4:
        for suffix, strip in [("ish", 2), ("ese", 3), ("ian", 2), ("ine", 2), ("an", 1)]:
            if term.endswith(suffix):
                base = term[:-len(suffix)]
                for candidate in [base, base + "a", base + "ia"]:
                    if _try(candidate):
                        _krovetz_cache[term] = candidate
                        return candidate

    # 4. Suffix stripping (approximate Krovetz for OOV terms)
    for suffix, replacement in _STRIP_RULES:
        if len(term) > len(suffix) + 3 and term.endswith(suffix):
            stem = term[: -len(suffix)] + replacement
            if _try(stem):
                _krovetz_cache[term] = stem
                return stem

    _krovetz_cache[term] = None
    return None


def process_query(text: str, pr=None) -> List[str]:
    """Tokenize, lowercase, remove INQUERY stops, apply Krovetz normalization.

    Terms that cannot be mapped to any in-index form are silently dropped,
    matching the behaviour of QL (which also skips OOV terms).
    """
    tokens = tokenize_string(text)
    filtered = [t for t in tokens if t and t not in INQUERY_STOPS]
    if pr is not None:
        normalized = [krovetz_approx(t, pr) for t in filtered]
        filtered = [t for t in normalized if t is not None]
    return filtered


# ── QL retrieval (count-only, fast) ──────────────────────────────────────────

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


# ── Positional data loading ───────────────────────────────────────────────────

def load_doc_positions(pr, term: str,
                       candidates: Optional[List[int]] = None) -> Dict[int, List[int]]:
    """Return {docid: [pos, ...]} for term.

    If *candidates* is provided (sorted list of docids), only reads positions
    for those documents — much faster for sparse candidate sets.
    """
    if candidates is not None:
        raw = pr.read_positions_for(term, candidates)
    else:
        raw = pr.read_positions(term)
    return {doc_id: list(positions) for doc_id, positions in raw}


# ── Bigram collection statistics (actual counts, not independence approx) ────

_bigram_col_stats: Dict[Tuple[str, str], Tuple[float, float]] = {}


def precompute_bigram_stats(queries: Dict[str, str], pr, C: int) -> None:
    """Scan full positional posting lists to get actual od/uw collection counts.

    The independence approximation (p(t1)*p(t2)) under-estimates real bigram
    frequencies by 100-1000x for meaningful phrases, making Dirichlet smoothing
    wildly over-weight the bigram features.  Actual counts fix this.

    Processes pairs grouped by first term so full position lists are streamed
    one term at a time and discarded after use.
    """
    from collections import defaultdict as _dd
    first_term_groups: Dict[str, set] = _dd(set)
    for qtext in queries.values():
        terms = process_query(qtext, pr)
        for i in range(len(terms) - 1):
            first_term_groups[terms[i]].add(terms[i + 1])

    n_pairs = sum(len(v) for v in first_term_groups.values())
    print(f"  Pre-computing {n_pairs} bigram collection stats …", flush=True)
    t0 = time.perf_counter()

    for t1, t2_set in sorted(first_term_groups.items()):
        pos1 = load_doc_positions(pr, t1)          # full collection for t1
        for t2 in sorted(t2_set):
            if (t1, t2) in _bigram_col_stats:
                continue
            pos2 = load_doc_positions(pr, t2)      # full collection for t2
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
    """Return (od_cf, uw_cf) for a term pair, using cached actual counts."""
    if (t1, t2) in _bigram_col_stats:
        return _bigram_col_stats[(t1, t2)]
    # Fallback if precompute was not called (graceful degradation)
    ind = term_cf.get(t1, 1e-9) * term_cf.get(t2, 1e-9)
    return max(ind, 1.0 / C), max(ind * 8, 8.0 / C)


# ── Bigram window counts ──────────────────────────────────────────────────────

def ordered_windows(pos1: List[int], pos2: List[int]) -> int:
    """Count occurrences of t2 appearing at exactly pos1+1 (ordered #od:1)."""
    if not pos1 or not pos2:
        return 0
    set2 = set(pos2)
    return sum(1 for p in pos1 if (p + 1) in set2)


def unordered_windows(pos1: List[int], pos2: List[int], win: int = 8) -> int:
    """Count positions in pos1 within `win` tokens of any position in pos2 (#uw:win)."""
    if not pos1 or not pos2:
        return 0
    count = 0
    j = 0
    sorted_pos2 = sorted(pos2)
    for p1 in sorted(pos1):
        # Advance pointer past positions too far left
        while j < len(sorted_pos2) and sorted_pos2[j] < p1 - win:
            j += 1
        # Check if any p2 in window
        k = j
        while k < len(sorted_pos2) and sorted_pos2[k] <= p1 + win:
            if sorted_pos2[k] != p1:
                count += 1
                break
            k += 1
    return count


# ── SDM (full, with bigram features) ─────────────────────────────────────────

def sdm_search(terms: List[str], pr, ls, C: int, N_DOCS: int,
               n: int = N, mu: float = MU,
               uni_w: float = SDM_UNI,
               od_w:  float = SDM_OD,
               uw_w:  float = SDM_UW) -> List[Tuple[int, float]]:
    """Sequential Dependence Model with ordered/unordered window bigrams."""
    terms = list(dict.fromkeys(terms))
    if not terms:
        return []

    # ── Phase 1: QL candidate retrieval (fast, count-only) ────────────────────
    candidates_scored = ql_search(terms, pr, ls, C, n=n, mu=mu)
    if not candidates_scored:
        return []
    candidate_set = {doc_id for doc_id, _ in candidates_scored}

    # ── Phase 2: Load positions for candidates only ───────────────────────────
    sorted_candidates = sorted(candidate_set)
    term_positions: Dict[str, Dict[int, List[int]]] = {}
    term_cf:        Dict[str, float] = {}
    # Only keep in-vocab terms (OOV terms corrupt scoring with spurious constants)
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

    # ── Phase 3: Score candidates with full SDM ───────────────────────────────
    scored: List[Tuple[int, float]] = []
    for doc_id in candidate_set:
        dl    = ls.length(doc_id)
        denom = dl + mu

        # Unigram component (in-vocab terms only)
        uni_score = sum(
            math.log((len(term_positions[t].get(doc_id, [])) + mu * term_cf[t]) / denom)
            for t in vocab_terms
        )

        # Bigram components (actual collection stats from precompute_bigram_stats)
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


# ── WSDM-Int (IDF-weighted unigrams + IDF-weighted bigrams) ──────────────────

def wsdm_int_search(terms: List[str], pr, ls, C: int, N_DOCS: int,
                    n: int = N, mu: float = MU,
                    uni_w: float = SDM_UNI,
                    od_w:  float = SDM_OD,
                    uw_w:  float = SDM_UW) -> List[Tuple[int, float]]:
    """WSDM-Internal: IDF-weighted unigrams + SDM-style bigrams.

    Approximation of Bendersky et al. 2010:
      - Unigrams: IDF-reweighted QL (normalised to sum to uni_w)
      - Bigrams: SDM fixed weights od_w / uw_w per adjacent pair
    This gives the IDF-reweighted unigrams of WSDM combined with the
    proximity features of SDM.
    """
    terms = list(dict.fromkeys(terms))
    if not terms:
        return []

    # Phase 1: QL candidate retrieval
    candidates_scored = ql_search(terms, pr, ls, C, n=n, mu=mu)
    if not candidates_scored:
        return []
    candidate_set = {doc_id for doc_id, _ in candidates_scored}

    # Phase 2: Load positions for candidates only (in-vocab terms only)
    sorted_candidates = sorted(candidate_set)
    term_positions: Dict[str, Dict[int, List[int]]] = {}
    term_cf: Dict[str, float] = {}
    term_df: Dict[str, int] = {}
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

    # IDF weights for unigrams
    def idf_fn(df: int) -> float:
        return math.log((N_DOCS + 1) / (df + 0.5)) if df > 0 else 0.0

    uni_idfs  = {t: idf_fn(term_df[t]) for t in vocab_terms}
    total_idf = sum(uni_idfs.values())
    if total_idf <= 0:
        return ql_search(terms, pr, ls, C, n=n, mu=mu)
    n_bigrams = max(len(vocab_terms) - 1, 1)

    # Phase 3+4: Compute bigram window counts and score candidates
    bigram_od: Dict[Tuple[Tuple[str,str], int], int] = {}
    bigram_uw: Dict[Tuple[Tuple[str,str], int], int] = {}

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
            score += (od_w / n_bigrams) * math.log(
                (od_tf + mu * od_bg) / denom)
            score += (uw_w / n_bigrams) * math.log(
                (uw_tf + mu * uw_bg) / denom)
        scored.append((doc_id, score))

    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── WSDM unigram-only (for comparison) ───────────────────────────────────────

def wsdm_unigram_search(terms: List[str], pr, ls, C: int, N_DOCS: int,
                        n: int = N, mu: float = MU) -> List[Tuple[int, float]]:
    """WSDM unigram-only approximation (IDF-weighted QL, no bigrams)."""
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
            for i, (p_t, idf, _) in enumerate(term_info)
        )
        scored.append((doc_id, score))
    scored.sort(key=lambda x: -x[1])
    return scored[:n]


# ── Main experiment ───────────────────────────────────────────────────────────

def run_experiment(queries_type: str = "titles", output_path: Optional[str] = None):
    idx, pr, ls, stats = load_index()
    C      = stats.collection_length
    N_DOCS = stats.total_document_count

    qfile   = TITLE_Q if queries_type == "titles" else DESC_Q
    queries = load_queries(qfile)
    qrels   = read_qrels(QRELS)
    print(f"Query set: {queries_type} ({len(queries)} queries)  "
          f"|  Judged topics: {len(qrels)}")
    print(f"Index: {PART}  (positions available)")

    precompute_bigram_stats(queries, pr, C)

    MODEL_NAMES = ["QL", "SDM", "WSDM-Uni", "WSDM-Int"]
    runs:   Dict[str, Dict] = {m: {} for m in MODEL_NAMES}
    timing: Dict[str, float] = {m: 0.0 for m in MODEL_NAMES}

    total = len(queries)
    print(f"\nRunning {total} queries × {len(MODEL_NAMES)} models …\n")

    for qi, (topic, query_text) in enumerate(sorted(queries.items()), 1):
        terms = process_query(query_text, pr)
        if qi % 50 == 1:
            print(f"  [{qi:3d}/{total}] topic {topic}: {query_text!r}",
                  flush=True)

        # QL
        t0 = time.perf_counter()
        res = ql_search(terms, pr, ls, C)
        timing["QL"] += time.perf_counter() - t0
        runs["QL"][topic] = [(idx.get_name(d), s) for d, s in res]

        # SDM (with ordered/unordered bigrams)
        t0 = time.perf_counter()
        res = sdm_search(terms, pr, ls, C, N_DOCS)
        timing["SDM"] += time.perf_counter() - t0
        runs["SDM"][topic] = [(idx.get_name(d), s) for d, s in res]

        # WSDM unigram (IDF-weighted QL, no bigrams — baseline)
        t0 = time.perf_counter()
        res = wsdm_unigram_search(terms, pr, ls, C, N_DOCS)
        timing["WSDM-Uni"] += time.perf_counter() - t0
        runs["WSDM-Uni"][topic] = [(idx.get_name(d), s) for d, s in res]

        # WSDM-Int (IDF-weighted unigrams + bigrams)
        t0 = time.perf_counter()
        res = wsdm_int_search(terms, pr, ls, C, N_DOCS)
        timing["WSDM-Int"] += time.perf_counter() - t0
        runs["WSDM-Int"][topic] = [(idx.get_name(d), s) for d, s in res]

    print("\nEvaluating …", flush=True)
    results: Dict[str, Dict[str, float]] = {}
    for model, run in runs.items():
        ranked = {t: [name for name, _ in docs] for t, docs in run.items()}
        results[model] = evaluate(ranked, qrels, metrics=METRICS)

    PAPER = {
        "titles": {"QL": 0.252, "SDM": 0.263, "WSDM-Int": 0.269},
        "descs":  {"QL": 0.244, "SDM": 0.258, "WSDM-Int": 0.278},
    }
    paper = PAPER.get(queries_type, {})

    lines = [f"# Robust04 SDM/WSDM-Int — {queries_type} (complete index with positions)\n"]
    lines.append(f"**Index:** `{PART}` (Krovetz, positional)  "
                 f"**Queries:** {len(queries)}  **μ:** {MU}\n")
    lines.append(f"**SDM weights:** uni={SDM_UNI}, od={SDM_OD}, uw={SDM_UW}\n\n")

    lines.append("## Results vs Paper\n")
    hdr = "| Model | MAP | NDCG@20 | P@20 | Paper MAP |"
    div = "|-------|-----|---------|------|-----------|"
    lines.append(hdr)
    lines.append(div)
    for m in MODEL_NAMES:
        r = results[m]
        p = paper.get(m, paper.get(m.replace("-Uni", "").replace("-Int", "Int"), None))
        # map model display name to paper key
        paper_map = {"QL": "QL", "SDM": "SDM", "WSDM-Uni": None, "WSDM-Int": "WSDM-Int"}
        p = paper.get(paper_map.get(m, ""), "-")
        p_str = f"{p:.3f}" if isinstance(p, float) else "—"
        lines.append(f"| {m:<9s} | {r['map']:.4f} | {r['ndcg@20']:.4f} | {r['p@20']:.4f} | {p_str} |")

    lines.append("")
    lines.append("## Timing\n")
    lines.append("| Model | Total (s) | Per query (ms) |")
    lines.append("|-------|-----------|----------------|")
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

    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Robust04 SDM/WSDM experiment with positions")
    ap.add_argument("--queries", choices=["titles", "descs"], default="titles")
    ap.add_argument("--output", default=None)
    args = ap.parse_args()
    if args.output is None:
        args.output = f"results/robust04_sdm_{args.queries}.md"
    run_experiment(args.queries, args.output)
